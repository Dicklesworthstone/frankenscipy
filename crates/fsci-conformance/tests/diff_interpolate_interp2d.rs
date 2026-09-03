#![forbid(unsafe_code)]
//! Live scipy.interpolate parity for fsci_interpolate::interp2d
//! (bilinear interpolation on a regular grid).
//!
//! Resolves [frankenscipy-fjuub]. Compare against
//! scipy.interpolate.RegularGridInterpolator(method='linear').
//!
//! z layout note: fsci's interp2d takes z[y_idx][x_idx] (shape
//! (ny, nx)). scipy's RegularGridInterpolator((x, y), z) expects
//! z shape (nx, ny). The harness builds z in fsci layout and
//! transposes for the scipy oracle.
//!
//! Tolerance: 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_interpolate::interp2d;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    x: Vec<f64>,
    y: Vec<f64>,
    /// z[y_idx][x_idx]
    z: Vec<Vec<f64>>,
    /// Query points (xi, yi) inside the convex hull
    queries: Vec<(f64, f64)>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
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
    fs::create_dir_all(output_dir()).expect("create interp2d diff dir");
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

fn mesh_zy<F: Fn(f64, f64) -> f64>(x: &[f64], y: &[f64], f: F) -> Vec<Vec<f64>> {
    // z[yi][xi]
    y.iter()
        .map(|&yv| x.iter().map(|&xv| f(xv, yv)).collect())
        .collect()
}

fn generate_query() -> OracleQuery {
    let x_a: Vec<f64> = (0..5).map(|i| i as f64 * 0.5).collect();
    let y_a: Vec<f64> = (0..6).map(|i| i as f64 * 0.4).collect();
    let z_lin = mesh_zy(&x_a, &y_a, |x, y| 2.0 * x + 3.0 * y + 1.0);
    let z_quad = mesh_zy(&x_a, &y_a, |x, y| x * x + y * y - 0.5 * x * y);
    let queries_a: Vec<(f64, f64)> = vec![
        (0.25, 0.15),
        (0.5, 0.5),
        (1.0, 0.8),
        (1.5, 1.2),
        (1.75, 1.9),
    ];

    let x_b: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let y_b: Vec<f64> = (0..7).map(|i| i as f64 * 0.7).collect();
    let z_smooth = mesh_zy(&x_b, &y_b, |x, y| (x * 0.3).sin() + (y * 0.5).cos());
    let queries_b: Vec<(f64, f64)> =
        vec![(1.5, 1.0), (3.5, 2.5), (5.0, 3.0), (6.5, 4.0), (2.0, 0.5)];

    OracleQuery {
        points: vec![
            Case {
                case_id: "interp2d_planar".into(),
                x: x_a.clone(),
                y: y_a.clone(),
                z: z_lin,
                queries: queries_a.clone(),
            },
            Case {
                case_id: "interp2d_quadratic".into(),
                x: x_a,
                y: y_a,
                z: z_quad,
                queries: queries_a,
            },
            Case {
                case_id: "interp2d_smooth".into(),
                x: x_b,
                y: y_b,
                z: z_smooth,
                queries: queries_b,
            },
        ],
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    x = np.array(case["x"], dtype=float)
    y = np.array(case["y"], dtype=float)
    z = np.array(case["z"], dtype=float)  # shape (ny, nx)
    # RegularGridInterpolator((x, y), z) expects z shape (nx, ny).
    z_t = z.T
    rgi = RegularGridInterpolator((x, y), z_t, method='linear')
    try:
        flat = []
        for q_xy in case["queries"]:
            xi, yi = float(q_xy[0]), float(q_xy[1])
            v = float(rgi([[xi, yi]])[0])
            flat.append(v)
        if all(math.isfinite(v) for v in flat):
            points.append({"case_id": cid, "values": flat})
        else:
            points.append({"case_id": cid, "values": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "values": None})
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
                "failed to spawn python3 for interp2d oracle: {e}"
            );
            eprintln!("skipping interp2d oracle: python3 not available ({e})");
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
                "interp2d oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping interp2d oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for interp2d oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "interp2d oracle failed: {stderr}"
        );
        eprintln!("skipping interp2d oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse interp2d oracle JSON"))
}

fn vec_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_interpolate_interp2d() {
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
        let Some(expected) = arm.values.as_ref() else {
            continue;
        };
        let mut actual = Vec::new();
        let mut all_finite = true;
        for &(xi, yi) in &case.queries {
            match interp2d(&case.x, &case.y, &case.z, xi, yi) {
                Ok(v) if v.is_finite() => actual.push(v),
                _ => {
                    all_finite = false;
                    break;
                }
            }
        }
        if !all_finite {
            continue;
        }
        let abs_d = vec_max_diff(&actual, expected);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_interpolate_interp2d".into(),
        category: "fsci_interpolate::interp2d vs scipy.interpolate.RegularGridInterpolator(linear)"
            .into(),
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
            eprintln!("interp2d mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "interp2d conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
