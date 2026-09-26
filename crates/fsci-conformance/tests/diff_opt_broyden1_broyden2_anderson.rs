#![forbid(unsafe_code)]
//! Live scipy.optimize.root parity for fsci_opt multivariate root finders
//! broyden1, broyden2, anderson.
//!
//! Resolves [frankenscipy-6rp8v]. Multivariate root systems may admit
//! multiple roots (e.g. lin_prod: x+y=3, xy=2 has roots (1,2) and (2,1));
//! different quasi-Newton variants legitimately converge to different
//! ones. The harness uses a property-based check: verify that BOTH fsci
//! and scipy solutions satisfy ||F(x)||∞ < 1e-5.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::{anderson, broyden1, broyden2};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-5;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    method: String,
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
    op: String,
    abs_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
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
    fs::create_dir_all(output_dir()).expect("create broyden diff dir");
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
        // F(x, y) = [x^2 + y - 3, x + y^2 - 3]; root near (1, ±1) and others
        "two_var_quad" => vec![x[0] * x[0] + x[1] - 3.0, x[0] + x[1] * x[1] - 3.0],
        // F(x, y) = [x + y - 3, x*y - 2]; roots (1, 2) and (2, 1)
        "lin_prod" => vec![x[0] + x[1] - 3.0, x[0] * x[1] - 2.0],
        // F(x, y, z) = [x + y + z - 6, x - y - 1, x * z - 6]; root (3, 2, 2)? Test: 3+2+2=7≠6
        // Try F = [x+y+z-6, x-y, x*y*z-8]; root (2, 2, 2)
        "cubic_3d" => vec![
            x[0] + x[1] + x[2] - 6.0,
            x[0] - x[1],
            x[0] * x[1] * x[2] - 8.0,
        ],
        _ => vec![],
    }
}

fn generate_query() -> OracleQuery {
    let probes: &[(&str, &str, Vec<f64>)] = &[
        ("broyden1", "two_var_quad", vec![1.5, 1.5]),
        ("broyden2", "two_var_quad", vec![1.5, 1.5]),
        ("anderson", "two_var_quad", vec![1.5, 1.5]),
        ("broyden1", "lin_prod", vec![1.5, 1.5]),
        ("broyden2", "lin_prod", vec![1.5, 1.5]),
        ("anderson", "lin_prod", vec![1.5, 1.5]),
        ("broyden1", "cubic_3d", vec![1.8, 1.8, 1.8]),
        ("broyden2", "cubic_3d", vec![1.8, 1.8, 1.8]),
    ];
    let points: Vec<Case> = probes
        .iter()
        .enumerate()
        .map(|(i, (method, fname, x0))| Case {
            case_id: format!("p{i:02}_{method}_{fname}"),
            method: (*method).into(),
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

def func(name, x):
    if name == "two_var_quad":
        return np.array([x[0]**2 + x[1] - 3.0, x[0] + x[1]**2 - 3.0])
    if name == "lin_prod":
        return np.array([x[0] + x[1] - 3.0, x[0]*x[1] - 2.0])
    if name == "cubic_3d":
        return np.array([x[0]+x[1]+x[2]-6.0, x[0]-x[1], x[0]*x[1]*x[2]-8.0])
    return None

method_map = {"broyden1": "broyden1", "broyden2": "broyden2", "anderson": "anderson"}

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    method = method_map.get(case["method"])
    fname = case["func"]
    x0 = np.array(case["x0"], dtype=float)
    try:
        sol = root(lambda x: func(fname, x), x0, method=method, tol=1e-12)
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
                "failed to spawn python3 for broyden oracle: {e}"
            );
            eprintln!("skipping broyden oracle: python3 not available ({e})");
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
                "broyden oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping broyden oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for broyden oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "broyden oracle failed: {stderr}"
        );
        eprintln!("skipping broyden oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse broyden oracle JSON"))
}

#[test]
fn diff_opt_broyden1_broyden2_anderson() {
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
    let methods = ["broyden1", "broyden2", "anderson"];
    let mut ledger = CompareLedger::new("diff_opt_broyden1_broyden2_anderson", &methods);

    for case in &query.points {
        // The oracle sends x only when SciPy's root converged.
        let expected = pmap.get(&case.case_id).and_then(|arm| arm.x.as_deref());
        let f = |x: &[f64]| func(&case.func, x);
        let res = match case.method.as_str() {
            "broyden1" => broyden1(f, &case.x0, 1.0e-10, 1000),
            "broyden2" => broyden2(f, &case.x0, 1.0e-10, 1000),
            "anderson" => anderson(f, &case.x0, 1.0e-10, 1000, 5, 1.0),
            other => panic!("unknown method {other} in {}", case.case_id),
        };
        // Not converging where SciPy converged is an fsci failure, not a skip.
        let rr = res.ok().filter(|rr| rr.converged);
        let Some((expected, fsci_x)) = ledger.slices(
            &case.method,
            &case.case_id,
            expected,
            rr.as_ref().map(|rr| rr.x.as_slice()),
        ) else {
            continue;
        };
        // Property-based: residual |F(x)| ≈ 0 for both fsci and scipy.
        let fsci_residual = func(&case.func, fsci_x)
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let scipy_residual = func(&case.func, expected)
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let abs_d = fsci_residual.max(scipy_residual);
        max_overall = max_overall.max(abs_d);
        ledger.compared(&case.method, &case.case_id, abs_d <= ABS_TOL);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.method.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_broyden1_broyden2_anderson".into(),
        category: "fsci_opt::{broyden1, broyden2, anderson} vs scipy.optimize.root".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
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
        "broyden/anderson conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(
        methods
            .iter()
            .map(|m| query.points.iter().filter(|c| c.method == *m).count())
            .min()
            .unwrap_or(0),
    );
}
