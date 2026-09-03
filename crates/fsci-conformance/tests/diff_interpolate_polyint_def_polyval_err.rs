#![forbid(unsafe_code)]
//! Live numpy parity for fsci_interpolate::{polyint_definite,
//! polyval_with_error}.
//!
//! Resolves [frankenscipy-2p4hz]. Both functions use HIGH-FIRST
//! polynomial coefficient convention.
//!   polyint_definite(coeffs, a, b) = ∫_a^b p(x) dx
//!   polyval_with_error returns (value, condition*ε); we compare the
//!   value only.
//!
//! Tolerance: 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_interpolate::{polyint_definite, polyval_with_error};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "polyint_def" | "polyval_err"
    coeffs: Vec<f64>,
    a: f64,
    b: f64,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
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
    fs::create_dir_all(output_dir()).expect("create polyint_def diff dir");
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
    // HIGH-FIRST coefficients.
    let p1 = vec![1.0_f64, 0.0]; // x
    let p2 = vec![1.0_f64, 0.0, 0.0]; // x^2
    let p3 = vec![1.0_f64, -2.0, 3.0]; // x^2 - 2x + 3
    let p4 = vec![0.5_f64, -1.0, 0.25, 2.0]; // 0.5x^3 - x^2 + 0.25x + 2

    let polys: Vec<(&str, Vec<f64>)> =
        vec![("x", p1), ("xsq", p2), ("x2_m2x_3", p3), ("cubic_4t", p4)];

    let mut points = Vec::new();
    let intervals = [(-1.0_f64, 1.0), (0.0, 2.0), (-2.0, 3.0)];
    let xs = [0.5_f64, 1.0, 2.5, -1.5];

    for (label, coeffs) in &polys {
        for &(a, b) in &intervals {
            points.push(Case {
                case_id: format!("polyint_def_{label}_{a}_{b}")
                    .replace('.', "p")
                    .replace('-', "n"),
                op: "polyint_def".into(),
                coeffs: coeffs.clone(),
                a,
                b,
                x: 0.0,
            });
        }
        for &x in &xs {
            points.push(Case {
                case_id: format!("polyval_err_{label}_x{x}")
                    .replace('.', "p")
                    .replace('-', "n"),
                op: "polyval_err".into(),
                coeffs: coeffs.clone(),
                a: 0.0,
                b: 0.0,
                x,
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
    coeffs = np.array(case["coeffs"], dtype=float)
    try:
        if op == "polyint_def":
            a = float(case["a"]); b = float(case["b"])
            antider = np.polyint(coeffs)
            v = float(np.polyval(antider, b) - np.polyval(antider, a))
        elif op == "polyval_err":
            v = float(np.polyval(coeffs, float(case["x"])))
        else:
            v = float("nan")
        if math.isfinite(v):
            points.append({"case_id": cid, "value": v})
        else:
            points.append({"case_id": cid, "value": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "value": None})
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
                "failed to spawn python3 for polyint_def oracle: {e}"
            );
            eprintln!("skipping polyint_def oracle: python3 not available ({e})");
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
                "polyint_def oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping polyint_def oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for polyint_def oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "polyint_def oracle failed: {stderr}"
        );
        eprintln!("skipping polyint_def oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse polyint_def oracle JSON"))
}

#[test]
fn diff_interpolate_polyint_def_polyval_err() {
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
        let Some(expected) = arm.value else {
            continue;
        };
        let actual = match case.op.as_str() {
            "polyint_def" => polyint_definite(&case.coeffs, case.a, case.b),
            "polyval_err" => polyval_with_error(&case.coeffs, case.x).0,
            _ => continue,
        };
        let abs_d = (actual - expected).abs();
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
        test_id: "diff_interpolate_polyint_def_polyval_err".into(),
        category: "fsci_interpolate::polyint_definite + polyval_with_error vs numpy".into(),
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
        "polyint_def/polyval_err conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
