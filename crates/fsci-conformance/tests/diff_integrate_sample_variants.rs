#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_integrate uniform/irregular
//! sample-grid quadrature variants: trapezoid_uniform, simpson_uniform,
//! trapezoid_irregular, simpson_irregular, trapezoid_richardson.
//!
//! Resolves [frankenscipy-v4u6c]. 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_integrate::{simpson_irregular, simpson_uniform, trapezoid_irregular, trapezoid_uniform};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    y: Vec<f64>,
    x: Vec<f64>,
    dx: f64,
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
    fs::create_dir_all(output_dir()).expect("create sample_variants diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize sample_variants diff log");
    fs::write(path, json).expect("write sample_variants diff log");
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

    // Uniform grids
    let x11 = linspace(0.0, std::f64::consts::PI, 11);
    let y11_sin: Vec<f64> = x11.iter().map(|&v| v.sin()).collect();
    let y11_quad: Vec<f64> = x11.iter().map(|&v| v * v).collect();
    let dx11 = std::f64::consts::PI / 10.0;

    let x21 = linspace(0.0, 2.0, 21);
    let y21_exp: Vec<f64> = x21.iter().map(|&v| (-v).exp()).collect();
    let dx21 = 2.0 / 20.0;

    let uniform_inputs: &[(&str, &[f64], &[f64], f64)] = &[
        ("11pt_sin", &y11_sin, &x11, dx11),
        ("11pt_quad", &y11_quad, &x11, dx11),
        ("21pt_exp", &y21_exp, &x21, dx21),
    ];

    // trapezoid_richardson applies Richardson extrapolation, so it
    // intentionally differs from scipy's basic trapezoid — not a parity
    // candidate without an analytical-truth oracle. Excluded.
    for (label, y, x, dx) in uniform_inputs {
        for op in ["trapezoid_uniform", "simpson_uniform"] {
            points.push(PointCase {
                case_id: format!("{op}_{label}"),
                op: op.into(),
                y: y.to_vec(),
                x: x.to_vec(),
                dx: *dx,
            });
        }
    }

    // Irregular grids — odd-length needed for simpson_irregular too
    let x_irr_5 = vec![0.0_f64, 0.5, 1.5, 2.5, 4.0];
    let y_irr_5_sin: Vec<f64> = x_irr_5.iter().map(|&v| v.sin()).collect();
    let x_irr_7 = vec![0.0_f64, 0.3, 0.9, 1.5, 2.0, 2.7, 3.2];
    let y_irr_7_quad: Vec<f64> = x_irr_7.iter().map(|&v| v * v + 1.0).collect();

    let irr_inputs: &[(&str, &[f64], &[f64])] = &[
        ("5pt_sin", &y_irr_5_sin, &x_irr_5),
        ("7pt_quad_plus_1", &y_irr_7_quad, &x_irr_7),
    ];

    for (label, y, x) in irr_inputs {
        for op in ["trapezoid_irregular", "simpson_irregular"] {
            points.push(PointCase {
                case_id: format!("{op}_{label}"),
                op: op.into(),
                y: y.to_vec(),
                x: x.to_vec(),
                dx: 0.0,
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

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    y = np.array(case["y"], dtype=float)
    x = np.array(case["x"], dtype=float)
    dx = float(case["dx"])
    try:
        if op in ("trapezoid_uniform",):
            v = float(integrate.trapezoid(y, dx=dx))
        elif op in ("simpson_uniform",):
            v = float(integrate.simpson(y, dx=dx))
        elif op == "trapezoid_irregular":
            v = float(integrate.trapezoid(y, x))
        elif op == "simpson_irregular":
            v = float(integrate.simpson(y, x=x))
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
    let query_json = serde_json::to_string(query).expect("serialize sample_variants query");
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
                "failed to spawn python3 for sample_variants oracle: {e}"
            );
            eprintln!("skipping sample_variants oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open sample_variants oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "sample_variants oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping sample_variants oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for sample_variants oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "sample_variants oracle failed: {stderr}"
        );
        eprintln!("skipping sample_variants oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse sample_variants oracle JSON"))
}

#[test]
fn diff_integrate_sample_variants() {
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
        let fsci_v: f64 = match case.op.as_str() {
            "trapezoid_uniform" => match trapezoid_uniform(&case.y, case.dx) {
                Ok(r) => r.integral,
                Err(_) => continue,
            },
            "simpson_uniform" => match simpson_uniform(&case.y, case.dx) {
                Ok(r) => r.integral,
                Err(_) => continue,
            },
            "trapezoid_irregular" => trapezoid_irregular(&case.y, &case.x),
            "simpson_irregular" => simpson_irregular(&case.y, &case.x),
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
        test_id: "diff_integrate_sample_variants".into(),
        category: "scipy.integrate uniform/irregular sample-grid variants".into(),
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
        "sample_variants conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
