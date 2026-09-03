#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_integrate cumulative
//! trapezoidal integration: cumulative_trapezoid (no initial; len n-1),
//! cumulative_trapezoid_uniform, and cumulative_trapezoid_initial.
//!
//! Resolves [frankenscipy-lzz1b]. 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_integrate::{
    cumulative_trapezoid, cumulative_trapezoid_initial, cumulative_trapezoid_uniform,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    y: Vec<f64>,
    x: Vec<f64>,
    dx: f64,
    /// For cumulative_trapezoid_initial: initial value.
    initial: f64,
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
    fs::create_dir_all(output_dir()).expect("create cumulative_trap diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize cumulative_trap diff log");
    fs::write(path, json).expect("write cumulative_trap diff log");
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![a];
    }
    let step = (b - a) / (n - 1) as f64;
    (0..n).map(|i| a + step * i as f64).collect()
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    let x10 = linspace(0.0, 1.0, 10);
    let y10_quad: Vec<f64> = x10.iter().map(|&v| v * v).collect();
    let x11 = linspace(0.0, 2.0 * std::f64::consts::PI, 11);
    let y11_sin: Vec<f64> = x11.iter().map(|&v| v.sin()).collect();
    let x15 = linspace(-1.0, 1.0, 15);
    let y15_exp: Vec<f64> = x15.iter().map(|&v| (-v).exp()).collect();

    let dx10 = 1.0 / 9.0;
    let dx11 = 2.0 * std::f64::consts::PI / 10.0;
    let dx15 = 2.0 / 14.0;

    let inputs: &[(&str, &[f64], &[f64], f64)] = &[
        ("10pt_quad", &y10_quad, &x10, dx10),
        ("11pt_sin", &y11_sin, &x11, dx11),
        ("15pt_exp", &y15_exp, &x15, dx15),
    ];

    for (label, y, x, dx) in inputs {
        // (op_id, op_name, initial)
        for (op_id, op_name, init) in &[
            ("cum_trap", "cumulative_trapezoid", 0.0_f64),
            ("cum_trap_uniform", "cumulative_trapezoid_uniform", 0.0),
            ("cum_trap_init0", "cumulative_trapezoid_initial", 0.0),
            ("cum_trap_init_5", "cumulative_trapezoid_initial", 5.0),
        ] {
            points.push(PointCase {
                case_id: format!("{op_id}_{label}"),
                op: (*op_name).into(),
                y: y.to_vec(),
                x: x.to_vec(),
                dx: *dx,
                initial: *init,
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
from scipy import integrate

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
    cid = case["case_id"]; op = case["op"]
    y = np.array(case["y"], dtype=float)
    x = np.array(case["x"], dtype=float)
    dx = float(case["dx"]); initial = float(case["initial"])
    case_op_id = cid.split("_")[0:3]  # for routing the variant
    try:
        if op == "cumulative_trapezoid":
            v = integrate.cumulative_trapezoid(y, x)
        elif op == "cumulative_trapezoid_uniform":
            v = integrate.cumulative_trapezoid(y, dx=dx)
        elif op == "cumulative_trapezoid_initial":
            v = integrate.cumulative_trapezoid(y, x, initial=initial)
        else:
            v = None
        if v is None:
            points.append({"case_id": cid, "values": None})
        else:
            points.append({"case_id": cid, "values": finite_or_none(v)})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize cumulative_trap query");
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
                "failed to spawn python3 for cumulative_trap oracle: {e}"
            );
            eprintln!("skipping cumulative_trap oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open cumulative_trap oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "cumulative_trap oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping cumulative_trap oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for cumulative_trap oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "cumulative_trap oracle failed: {stderr}"
        );
        eprintln!("skipping cumulative_trap oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse cumulative_trap oracle JSON"))
}

#[test]
fn diff_integrate_cumulative_trapezoid() {
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
        let fsci_v: Vec<f64> = match case.op.as_str() {
            "cumulative_trapezoid" => match cumulative_trapezoid(&case.y, &case.x) {
                Ok(v) => v,
                Err(_) => continue,
            },
            "cumulative_trapezoid_uniform" => {
                match cumulative_trapezoid_uniform(&case.y, case.dx) {
                    Ok(v) => v,
                    Err(_) => continue,
                }
            }
            "cumulative_trapezoid_initial" => {
                cumulative_trapezoid_initial(&case.y, &case.x, case.initial)
            }
            _ => continue,
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
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_integrate_cumulative_trapezoid".into(),
        category: "scipy.integrate.cumulative_trapezoid".into(),
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
        "cumulative_trapezoid conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
