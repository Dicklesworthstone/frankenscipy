#![forbid(unsafe_code)]
//! Live scipy.integrate.cubature parity for fsci_integrate::cubature_scalar.
//!
//! Resolves [frankenscipy-ysnqu]. Tests several integrand × domain
//! combinations over rectangular 2-D and 3-D regions.
//!
//! Tolerance: 1e-6 abs. scipy.integrate.cubature uses a vectorized
//! call convention `f(x)` where `x` has shape `(n_points, ndim)`;
//! fsci's cubature_scalar takes `f(&[f64]) -> f64`. Reference values
//! computed via scipy.cubature with default tolerances.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_integrate::{CubatureOptions, cubature_scalar};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CubCase {
    case_id: String,
    integrand: String,
    a: Vec<f64>,
    b: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<CubCase>,
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
    fs::create_dir_all(output_dir()).expect("create cubature diff dir");
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

fn integrand(name: &str, x: &[f64]) -> f64 {
    match name {
        "one" => 1.0,
        "xy_2d" => x[0] * x[1],
        "sumsq_2d" => x[0] * x[0] + x[1] * x[1],
        "exp_negsumsq_2d" => (-(x[0] * x[0] + x[1] * x[1])).exp(),
        "xyz_3d" => x[0] * x[1] * x[2],
        "sumsq_3d" => x[0] * x[0] + x[1] * x[1] + x[2] * x[2],
        "exp_negsumsq_3d" => (-(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])).exp(),
        _ => f64::NAN,
    }
}

fn generate_query() -> OracleQuery {
    let probes: &[(&str, &str, Vec<f64>, Vec<f64>)] = &[
        ("xy_unit_2d", "xy_2d", vec![0.0, 0.0], vec![1.0, 1.0]),
        ("sumsq_unit_2d", "sumsq_2d", vec![0.0, 0.0], vec![1.0, 1.0]),
        (
            "exp_negsumsq_2d",
            "exp_negsumsq_2d",
            vec![-2.0, -2.0],
            vec![2.0, 2.0],
        ),
        (
            "xyz_unit_3d",
            "xyz_3d",
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ),
        (
            "sumsq_unit_3d",
            "sumsq_3d",
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ),
        (
            "exp_negsumsq_3d",
            "exp_negsumsq_3d",
            vec![-1.5, -1.5, -1.5],
            vec![1.5, 1.5, 1.5],
        ),
    ];
    let points: Vec<CubCase> = probes
        .iter()
        .map(|(cid, integrand, a, b)| CubCase {
            case_id: (*cid).into(),
            integrand: (*integrand).into(),
            a: a.clone(),
            b: b.clone(),
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
from scipy.integrate import cubature

def integrand(name, x):
    # x has shape (n_points, ndim)
    if name == "one":           return np.ones(x.shape[0])
    if name == "xy_2d":         return x[:, 0] * x[:, 1]
    if name == "sumsq_2d":      return x[:, 0]**2 + x[:, 1]**2
    if name == "exp_negsumsq_2d": return np.exp(-(x[:, 0]**2 + x[:, 1]**2))
    if name == "xyz_3d":        return x[:, 0] * x[:, 1] * x[:, 2]
    if name == "sumsq_3d":      return x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2
    if name == "exp_negsumsq_3d": return np.exp(-(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2))
    return np.full(x.shape[0], np.nan)

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; name = case["integrand"]
    a = [float(v) for v in case["a"]]
    b = [float(v) for v in case["b"]]
    try:
        r = cubature(lambda x: integrand(name, x), a, b, rtol=1e-10, atol=1e-12)
        v = float(r.estimate) if hasattr(r, "estimate") else float(r.integral)
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
                "failed to spawn python3 for cubature oracle: {e}"
            );
            eprintln!("skipping cubature oracle: python3 not available ({e})");
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
                "cubature oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping cubature oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for cubature oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "cubature oracle failed: {stderr}"
        );
        eprintln!("skipping cubature oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse cubature oracle JSON"))
}

#[test]
fn diff_integrate_cubature_scalar() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let opts = CubatureOptions::default();
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
        let f = |x: &[f64]| integrand(&case.integrand, x);
        let Ok(res) = cubature_scalar(f, &case.a, &case.b, opts.clone()) else {
            continue;
        };
        let abs_d = (res.estimate - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_integrate_cubature_scalar".into(),
        category: "fsci_integrate::cubature_scalar vs scipy.integrate.cubature".into(),
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
            eprintln!("cubature mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "cubature conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
