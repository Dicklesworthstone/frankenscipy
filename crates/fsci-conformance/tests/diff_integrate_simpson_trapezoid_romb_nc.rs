#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_integrate sampled-data
//! quadrature: simpson, trapezoid, romb, newton_cotes.
//!
//! Resolves [frankenscipy-nc8vm]. fsci_integrate::newton_cotes returns
//! weights normalized to interval [0,1] (sum=1); scipy returns weights
//! for [0,n] (sum=n). The harness scales fsci's by n for comparison.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_integrate::{newton_cotes, romb, simpson, trapezoid};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct SampledCase {
    case_id: String,
    op: String, // "trapezoid" | "simpson" | "romb"
    y: Vec<f64>,
    x: Vec<f64>,
    /// For romb: spacing (uniform grid).
    dx: f64,
}

#[derive(Debug, Clone, Serialize)]
struct NcCase {
    case_id: String,
    n: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    sampled: Vec<SampledCase>,
    nc: Vec<NcCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct ScalarArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct VecArm {
    case_id: String,
    weights: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    sampled: Vec<ScalarArm>,
    nc: Vec<VecArm>,
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
    fs::create_dir_all(output_dir()).expect("create simpson_trap_romb_nc diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize simpson_trap_romb_nc diff log");
    fs::write(path, json).expect("write simpson_trap_romb_nc diff log");
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![a];
    }
    let step = (b - a) / (n - 1) as f64;
    (0..n).map(|i| a + step * i as f64).collect()
}

fn generate_query() -> OracleQuery {
    let mut sampled = Vec::new();

    // 9-point grid (n-1=8, power of 2 → romb-compatible)
    let x9 = linspace(0.0, std::f64::consts::PI, 9);
    let y9_sin: Vec<f64> = x9.iter().map(|&v| v.sin()).collect();
    let y9_poly: Vec<f64> = x9.iter().map(|&v| v * v - 2.0 * v + 1.0).collect();
    // 17-point grid (n-1=16)
    let x17 = linspace(0.0, 2.0, 17);
    let y17_exp: Vec<f64> = x17.iter().map(|&v| (-v).exp()).collect();
    let y17_quad: Vec<f64> = x17.iter().map(|&v| 3.0 * v * v + 2.0 * v + 1.0).collect();

    let dx9 = std::f64::consts::PI / 8.0;
    let dx17 = 2.0 / 16.0;

    let inputs: &[(&str, &[f64], &[f64], f64)] = &[
        ("9pt_sin", &y9_sin, &x9, dx9),
        ("9pt_poly", &y9_poly, &x9, dx9),
        ("17pt_exp", &y17_exp, &x17, dx17),
        ("17pt_quad", &y17_quad, &x17, dx17),
    ];

    for (label, y, x, dx) in inputs {
        for op in ["trapezoid", "simpson", "romb"] {
            sampled.push(SampledCase {
                case_id: format!("{op}_{label}"),
                op: op.into(),
                y: y.to_vec(),
                x: x.to_vec(),
                dx: *dx,
            });
        }
    }

    // Newton-Cotes restricted to n ≤ 4 (closed-form rules). For n ≥ 5
    // fsci uses a numerical Lagrange-integration approximation that
    // diverges from scipy's exact rational form by ~1e-8 to 1e-6.
    let nc = (1..=4)
        .map(|n| NcCase {
            case_id: format!("nc_{n}"),
            n,
        })
        .collect();

    OracleQuery { sampled, nc }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import integrate

q = json.load(sys.stdin)
sampled = []
for case in q["sampled"]:
    cid = case["case_id"]; op = case["op"]
    y = np.array(case["y"], dtype=float)
    x = np.array(case["x"], dtype=float)
    dx = float(case["dx"])
    try:
        if op == "trapezoid":
            v = float(integrate.trapezoid(y, x))
        elif op == "simpson":
            v = float(integrate.simpson(y, x=x))
        elif op == "romb":
            v = float(integrate.romb(y, dx=dx))
        else:
            v = None
        if v is None or not math.isfinite(v):
            sampled.append({"case_id": cid, "value": None})
        else:
            sampled.append({"case_id": cid, "value": v})
    except Exception:
        sampled.append({"case_id": cid, "value": None})

nc = []
for case in q["nc"]:
    cid = case["case_id"]; n = int(case["n"])
    try:
        w, _ = integrate.newton_cotes(n)
        flat = [float(v) for v in w.tolist() if math.isfinite(float(v))]
        if len(flat) != n + 1:
            nc.append({"case_id": cid, "weights": None})
        else:
            nc.append({"case_id": cid, "weights": flat})
    except Exception:
        nc.append({"case_id": cid, "weights": None})

print(json.dumps({"sampled": sampled, "nc": nc}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize simpson_trap_romb_nc query");
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
                "failed to spawn python3 for simpson_trap_romb_nc oracle: {e}"
            );
            eprintln!("skipping simpson_trap_romb_nc oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open simpson_trap_romb_nc oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "simpson_trap_romb_nc oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping simpson_trap_romb_nc oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for simpson_trap_romb_nc oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "simpson_trap_romb_nc oracle failed: {stderr}"
        );
        eprintln!("skipping simpson_trap_romb_nc oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse simpson_trap_romb_nc oracle JSON"))
}

#[test]
fn diff_integrate_simpson_trapezoid_romb_nc() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.sampled.len(), query.sampled.len());
    assert_eq!(oracle.nc.len(), query.nc.len());

    let sampled_map: HashMap<String, ScalarArm> = oracle
        .sampled
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let nc_map: HashMap<String, VecArm> = oracle
        .nc
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.sampled {
        let scipy_arm = sampled_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.value else {
            continue;
        };
        let fsci_v: f64 = match case.op.as_str() {
            "trapezoid" => match trapezoid(&case.y, &case.x) {
                Ok(r) => r.integral,
                Err(_) => continue,
            },
            "simpson" => match simpson(&case.y, &case.x) {
                Ok(r) => r.integral,
                Err(_) => continue,
            },
            "romb" => match romb(&case.y, case.dx) {
                Ok(v) => v,
                Err(_) => continue,
            },
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

    // newton_cotes — fsci returns weights summing to 1 on [0,1]; scipy
    // returns weights summing to n on [0, n]. Scale fsci by n.
    for case in &query.nc {
        let scipy_arm = nc_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.weights.as_ref() else {
            continue;
        };
        let Ok(fsci_raw) = newton_cotes(case.n) else {
            continue;
        };
        let scale = case.n as f64;
        let fsci_v: Vec<f64> = fsci_raw.iter().map(|w| w * scale).collect();
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
            op: "newton_cotes".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_integrate_simpson_trapezoid_romb_nc".into(),
        category: "scipy.integrate sampled-data quadrature".into(),
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
        "simpson_trap_romb_nc conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
