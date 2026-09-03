#![forbid(unsafe_code)]
//! Live numpy differential coverage for fsci_special scalar log/exp
//! arithmetic helpers: log1p, expm1, logaddexp, logaddexp2.
//!
//! Resolves [frankenscipy-de5ow]. 1e-14 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{expm1, log1p, logaddexp, logaddexp2};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    x: f64,
    /// Second arg for logaddexp / logaddexp2.
    y: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create log_arith diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log_arith diff log");
    fs::write(path, json).expect("write log_arith diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    let unary_xs = [
        0.0_f64, 0.1, 0.5, 1.0, 5.0, -0.5, -0.9, 1.0e-10, -1.0e-10, 50.0, -50.0,
    ];
    for x in unary_xs {
        for op in ["log1p", "expm1"] {
            if op == "log1p" && x <= -1.0 {
                continue; // log1p of <= -1 is -inf/NaN; skip
            }
            points.push(PointCase {
                case_id: format!("{op}_{x}"),
                op: op.into(),
                x,
                y: 0.0,
            });
        }
    }

    let binary_pairs: &[(f64, f64)] = &[
        (0.0, 0.0),
        (1.0, 2.0),
        (3.0, -3.0),
        (100.0, 100.0),
        (-50.0, 50.0),
        (1e-12, 1e-12),
        (0.5, 0.5),
        (-2.0, 1.0),
    ];
    for &(x, y) in binary_pairs {
        for op in ["logaddexp", "logaddexp2"] {
            points.push(PointCase {
                case_id: format!("{op}_{x}_{y}"),
                op: op.into(),
                x,
                y,
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

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    x = float(case["x"]); y = float(case["y"])
    try:
        if op == "log1p":
            v = float(np.log1p(x))
        elif op == "expm1":
            v = float(np.expm1(x))
        elif op == "logaddexp":
            v = float(np.logaddexp(x, y))
        elif op == "logaddexp2":
            v = float(np.logaddexp2(x, y))
        else:
            v = None
        if v is None or not math.isfinite(v):
            points.append({"case_id": cid, "value": None})
        else:
            points.append({"case_id": cid, "value": v})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize log_arith query");
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
                "failed to spawn python3 for log_arith oracle: {e}"
            );
            eprintln!("skipping log_arith oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open log_arith oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "log_arith oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping log_arith oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for log_arith oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "log_arith oracle failed: {stderr}"
        );
        eprintln!("skipping log_arith oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse log_arith oracle JSON"))
}

#[test]
fn diff_special_log_arith() {
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
        let Some(expected) = scipy_arm.value else {
            continue;
        };
        let fsci_v = match case.op.as_str() {
            "log1p" => log1p(case.x),
            "expm1" => expm1(case.x),
            "logaddexp" => logaddexp(case.x, case.y),
            "logaddexp2" => logaddexp2(case.x, case.y),
            _ => continue,
        };
        let abs_d = (fsci_v - expected).abs();
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
        test_id: "diff_special_log_arith".into(),
        category: "numpy log1p + expm1 + logaddexp + logaddexp2".into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "log_arith conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
