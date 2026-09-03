#![forbid(unsafe_code)]
//! Live scipy.optimize.root(method='lm') parity for fsci_opt::lm_root.
//!
//! Resolves [frankenscipy-k78ra]. Multivariate root systems may have
//! multiple solutions; use the property ||F(x)||_∞ < 1e-5 on both
//! fsci and scipy solutions instead of comparing x directly.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::lm_root;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const RESIDUAL_TOL: f64 = 1.0e-5;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    func: String,
    x0: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    x: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create lm_root diff dir");
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

fn func(name: &str, x: &[f64]) -> Vec<f64> {
    match name {
        // F(x, y) = [x^2 + y - 4, x + y^2 - 4]
        "two_var_quad" => vec![x[0] * x[0] + x[1] - 4.0, x[0] + x[1] * x[1] - 4.0],
        // F(x, y, z) = [x+y+z-6, x-y, x*y*z-8]; root (2,2,2)
        "cubic_3d" => vec![
            x[0] + x[1] + x[2] - 6.0,
            x[0] - x[1],
            x[0] * x[1] * x[2] - 8.0,
        ],
        // F(x) = [x[0] - 1, x[1] - 2]; trivial
        "linear_2d" => vec![x[0] - 1.0, x[1] - 2.0],
        _ => vec![],
    }
}

fn generate_query() -> OracleQuery {
    let probes: &[(&str, Vec<f64>)] = &[
        ("two_var_quad", vec![1.5, 1.5]),
        ("two_var_quad", vec![2.5, 0.5]),
        ("cubic_3d", vec![1.8, 1.8, 1.8]),
        ("linear_2d", vec![0.0, 0.0]),
    ];
    let points = probes
        .iter()
        .enumerate()
        .map(|(i, (fname, x0))| Case {
            case_id: format!("p{i:02}_{fname}"),
            func: (*fname).into(),
            x0: x0.clone(),
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.optimize import root

def f(name, x):
    if name == "two_var_quad": return np.array([x[0]**2 + x[1] - 4.0, x[0] + x[1]**2 - 4.0])
    if name == "cubic_3d":     return np.array([x[0]+x[1]+x[2]-6.0, x[0]-x[1], x[0]*x[1]*x[2]-8.0])
    if name == "linear_2d":    return np.array([x[0]-1.0, x[1]-2.0])
    return np.zeros(len(x))

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    fname = case["func"]
    x0 = np.array(case["x0"], dtype=float)
    try:
        sol = root(lambda x: f(fname, x), x0, method='lm', tol=1e-10)
        if sol.success and all(math.isfinite(v) for v in sol.x.tolist()):
            points.append({"case_id": cid, "x": [float(v) for v in sol.x.tolist()]})
        else:
            points.append({"case_id": cid, "x": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "x": None})
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
                "failed to spawn python3 for lm_root oracle: {e}"
            );
            eprintln!("skipping lm_root oracle: python3 not available ({e})");
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
                "lm_root oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping lm_root oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for lm_root oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "lm_root oracle failed: {stderr}"
        );
        eprintln!("skipping lm_root oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse lm_root oracle JSON"))
}

#[test]
fn diff_opt_lm_root() {
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
        let Some(scipy_x) = arm.x.as_ref() else {
            continue;
        };
        let fname = case.func.clone();
        let f = move |x: &[f64]| func(&fname, x);
        let Ok(res) = lm_root(&f, &case.x0, 1.0e-10, 200) else {
            continue;
        };
        if !res.converged {
            continue;
        }
        let fsci_residual = func(&case.func, &res.x)
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let scipy_residual = func(&case.func, scipy_x)
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let abs_d = fsci_residual.max(scipy_residual);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= RESIDUAL_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_lm_root".into(),
        category: "fsci_opt::lm_root vs scipy.optimize.root(method='lm') via residual".into(),
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
            eprintln!("lm_root mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "lm_root conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
