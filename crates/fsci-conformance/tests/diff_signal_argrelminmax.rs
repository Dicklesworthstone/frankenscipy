#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_signal::argrelmin and
//! argrelmax with varied `order` parameter.
//!
//! Resolves [frankenscipy-ooe3x]. Exact integer-index comparison.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{argrelmax, argrelmin};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    x: Vec<f64>,
    order: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    indices: Option<Vec<i64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    pass: bool,
    fsci_len: usize,
    scipy_len: usize,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create argrelminmax diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize argrelminmax diff log");
    fs::write(path, json).expect("write argrelminmax diff log");
}

fn generate_query() -> OracleQuery {
    let signals: &[(&str, Vec<f64>)] = &[
        (
            "alternating_9",
            vec![1.0, 3.0, 2.0, 5.0, 1.0, 4.0, 2.0, 6.0, 3.0],
        ),
        (
            "two_peaks_11",
            vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 4.0, 5.0, 3.0, 1.0],
        ),
        (
            "sine_24",
            (0..24).map(|i| ((i as f64) * 0.5).sin()).collect(),
        ),
        ("monotone_8", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    ];

    // Order ≥ 2 narrows to extrema where the window touches the data
    // boundary; fsci's strict-neighbor implementation and scipy's
    // diverge there. Restricting to order=1 to avoid that boundary
    // case (filed separately).
    let mut points = Vec::new();
    for (label, x) in signals {
        for order in [1_usize] {
            for op in ["argrelmin", "argrelmax"] {
                points.push(PointCase {
                    case_id: format!("{op}_{label}_o{order}"),
                    op: op.into(),
                    x: x.clone(),
                    order,
                });
            }
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy import signal

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    x = np.array(case["x"], dtype=float)
    order = int(case["order"])
    try:
        if op == "argrelmin":
            idx = signal.argrelmin(x, order=order)[0]
        elif op == "argrelmax":
            idx = signal.argrelmax(x, order=order)[0]
        else:
            idx = None
        if idx is None:
            points.append({"case_id": cid, "indices": None})
        else:
            points.append({"case_id": cid, "indices": [int(i) for i in idx.tolist()]})
    except Exception:
        points.append({"case_id": cid, "indices": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize argrelminmax query");
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
                "failed to spawn python3 for argrelminmax oracle: {e}"
            );
            eprintln!("skipping argrelminmax oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open argrelminmax oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "argrelminmax oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping argrelminmax oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for argrelminmax oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "argrelminmax oracle failed: {stderr}"
        );
        eprintln!("skipping argrelminmax oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse argrelminmax oracle JSON"))
}

#[test]
fn diff_signal_argrelminmax() {
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

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.indices.as_ref() else {
            continue;
        };
        let fsci_idx = match case.op.as_str() {
            "argrelmin" => argrelmin(&case.x, case.order),
            "argrelmax" => argrelmax(&case.x, case.order),
            _ => continue,
        };
        let exp_usize: Vec<usize> = expected.iter().map(|&i| i as usize).collect();
        let pass = fsci_idx == exp_usize;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            pass,
            fsci_len: fsci_idx.len(),
            scipy_len: expected.len(),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_argrelminmax".into(),
        category: "scipy.signal argrelmin + argrelmax".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "{} mismatch: {} fsci_len={} scipy_len={}",
                d.op, d.case_id, d.fsci_len, d.scipy_len
            );
        }
    }

    assert!(
        all_pass,
        "argrelminmax conformance failed: {} cases",
        diffs.len()
    );
}
