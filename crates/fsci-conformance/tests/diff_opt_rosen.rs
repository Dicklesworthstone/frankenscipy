#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Rosenbrock test
//! function and its gradient:
//!   - `scipy.optimize.rosen(x)`     → Σ 100·(x[i+1] − x[i]²)² + (1 − x[i])²
//!   - `scipy.optimize.rosen_der(x)` → analytic gradient
//!
//! Resolves [frankenscipy-l0kd2]. These are reference test
//! functions for nonlinear optimisers, but fsci_opt::rosen /
//! rosen_der had no dedicated conformance harness — coverage was
//! only through `minimize`'s use of rosenbrock2 fixture cases in
//! P2C-003. Both are closed-form polynomials, so machine-precision
//! agreement is the right bar.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::{rosen, rosen_der};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-003";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    rosen: Option<f64>,
    der: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    arm: String,
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
    fs::create_dir_all(output_dir()).expect("create rosen diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize rosen diff log");
    fs::write(path, json).expect("write rosen diff log");
}

fn generate_query() -> OracleQuery {
    // Probe vectors: minimum (rosen=0), zero, classic Powell start,
    // 3D and 4D minima, off-minimum mixed.
    let inputs: &[(&str, Vec<f64>)] = &[
        ("2d_minimum", vec![1.0, 1.0]),
        ("2d_zero", vec![0.0, 0.0]),
        ("2d_powell_start", vec![-1.2, 1.0]),
        ("2d_off_axis", vec![2.0, 4.0]),
        ("3d_minimum", vec![1.0, 1.0, 1.0]),
        ("3d_mixed", vec![-1.0, 0.5, 2.0]),
        ("4d_minimum", vec![1.0, 1.0, 1.0, 1.0]),
        ("4d_uniform", vec![0.5, 0.5, 0.5, 0.5]),
        ("5d_perturbed", vec![0.9, 1.1, 0.8, 1.2, 0.95]),
    ];
    let points = inputs
        .iter()
        .map(|(name, x)| PointCase {
            case_id: (*name).into(),
            x: x.clone(),
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
from scipy import optimize

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

def vec_or_none(arr):
    out = []
    for v in arr.tolist():
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
    x = np.array(case["x"], dtype=float)
    try:
        points.append({
            "case_id": cid,
            "rosen": fnone(optimize.rosen(x)),
            "der": vec_or_none(optimize.rosen_der(x)),
        })
    except Exception:
        points.append({"case_id": cid, "rosen": None, "der": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize rosen query");
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
                "failed to spawn python3 for rosen oracle: {e}"
            );
            eprintln!("skipping rosen oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open rosen oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "rosen oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping rosen oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for rosen oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "rosen oracle failed: {stderr}"
        );
        eprintln!("skipping rosen oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse rosen oracle JSON"))
}

#[test]
fn diff_opt_rosen() {
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
        let fsci_r = rosen(&case.x);
        if let Some(scipy_r) = scipy_arm.rosen {
            let abs_d = (fsci_r - scipy_r).abs();
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "rosen".into(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }
        let fsci_der = rosen_der(&case.x);
        if let Some(scipy_der) = scipy_arm.der.as_ref()
            && fsci_der.len() == scipy_der.len()
        {
            let abs_d = fsci_der
                .iter()
                .zip(scipy_der.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "rosen_der".into(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_rosen".into(),
        category: "scipy.optimize.rosen / rosen_der".into(),
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
                "rosen {} mismatch: {} abs_diff={}",
                d.arm, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.optimize.rosen/rosen_der conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
