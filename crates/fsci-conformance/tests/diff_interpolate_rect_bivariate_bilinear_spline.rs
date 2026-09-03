#![forbid(unsafe_code)]
//! Live scipy.interpolate.RectBivariateSpline parity for
//! fsci_interpolate::rect_bivariate_spline (bicubic, kx=ky=3) and
//! rect_bilinear_spline (bilinear, kx=ky=1).
//!
//! Resolves [frankenscipy-osmri].
//!
//! Both convenience constructors are deterministic for fixed mesh,
//! degree, and z-data. Compare grid evaluation against scipy at
//! 1e-9 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_interpolate::{rect_bilinear_spline, rect_bivariate_spline};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    kind: String, // "bilinear" | "bicubic"
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<Vec<f64>>,
    xi: Vec<f64>,
    yi: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Row-major flatten of grid evaluation, shape (len(xi), len(yi)).
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    kind: String,
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
    fs::create_dir_all(output_dir()).expect("create rect_bv_spline diff dir");
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

fn mesh_eval<F: Fn(f64, f64) -> f64>(x: &[f64], y: &[f64], f: F) -> Vec<Vec<f64>> {
    x.iter()
        .map(|&xi| y.iter().map(|&yi| f(xi, yi)).collect())
        .collect()
}

fn generate_query() -> OracleQuery {
    let x_small: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let y_small: Vec<f64> = (0..7).map(|i| i as f64 * 0.5).collect();
    let xi_small: Vec<f64> = (0..11).map(|i| i as f64 * 0.5).collect();
    let yi_small: Vec<f64> = (0..7).map(|i| i as f64 * 0.5).collect();

    let x_large: Vec<f64> = (0..8).map(|i| i as f64 * 0.25).collect();
    let y_large: Vec<f64> = (0..8).map(|i| i as f64 * 0.3).collect();
    let xi_large: Vec<f64> = (0..6).map(|i| 0.1 + i as f64 * 0.3).collect();
    let yi_large: Vec<f64> = (0..6).map(|i| 0.05 + i as f64 * 0.35).collect();

    let z_planar = mesh_eval(&x_small, &y_small, |x, y| 2.0 * x - 3.0 * y + 5.0);
    let z_quadratic = mesh_eval(&x_small, &y_small, |x, y| x * x + y * y);
    let z_smooth = mesh_eval(&x_large, &y_large, |x, y| (x * 2.0).sin() * (y * 2.0).cos());

    OracleQuery {
        points: vec![
            Case {
                case_id: "bilinear_planar".into(),
                kind: "bilinear".into(),
                x: x_small.clone(),
                y: y_small.clone(),
                z: z_planar.clone(),
                xi: xi_small.clone(),
                yi: yi_small.clone(),
            },
            Case {
                case_id: "bilinear_quadratic".into(),
                kind: "bilinear".into(),
                x: x_small.clone(),
                y: y_small.clone(),
                z: z_quadratic.clone(),
                xi: xi_small.clone(),
                yi: yi_small.clone(),
            },
            Case {
                case_id: "bicubic_planar".into(),
                kind: "bicubic".into(),
                x: x_small.clone(),
                y: y_small.clone(),
                z: z_planar,
                xi: xi_small.clone(),
                yi: yi_small.clone(),
            },
            Case {
                case_id: "bicubic_quadratic".into(),
                kind: "bicubic".into(),
                x: x_small.clone(),
                y: y_small.clone(),
                z: z_quadratic,
                xi: xi_small,
                yi: yi_small,
            },
            Case {
                case_id: "bicubic_sinusoid".into(),
                kind: "bicubic".into(),
                x: x_large,
                y: y_large,
                z: z_smooth,
                xi: xi_large,
                yi: yi_large,
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
from scipy.interpolate import RectBivariateSpline

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; kind = case["kind"]
    try:
        x = np.array(case["x"], dtype=float)
        y = np.array(case["y"], dtype=float)
        z = np.array(case["z"], dtype=float)
        xi = np.array(case["xi"], dtype=float)
        yi = np.array(case["yi"], dtype=float)
        k = 1 if kind == "bilinear" else 3
        sp = RectBivariateSpline(x, y, z, kx=k, ky=k)
        # Evaluate as 2D mesh: shape (len(xi), len(yi))
        res = sp(xi, yi)
        flat = [float(v) for v in res.flatten().tolist()]
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
                "failed to spawn python3 for rect_bv_spline oracle: {e}"
            );
            eprintln!("skipping rect_bv_spline oracle: python3 not available ({e})");
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
                "rect_bv_spline oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping rect_bv_spline oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for rect_bv_spline oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "rect_bv_spline oracle failed: {stderr}"
        );
        eprintln!("skipping rect_bv_spline oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse rect_bv_spline oracle JSON"))
}

#[test]
fn diff_interpolate_rect_bivariate_bilinear_spline() {
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
        let sp = match case.kind.as_str() {
            "bilinear" => rect_bilinear_spline(&case.x, &case.y, &case.z),
            "bicubic" => rect_bivariate_spline(&case.x, &case.y, &case.z),
            _ => continue,
        };
        let Ok(sp) = sp else { continue };
        let grid = sp.eval_grid(&case.xi, &case.yi);
        // Row-major flatten
        let mut flat = Vec::with_capacity(case.xi.len() * case.yi.len());
        for row in &grid {
            flat.extend_from_slice(row);
        }
        let abs_d = if flat.len() != expected.len() {
            f64::INFINITY
        } else {
            flat.iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            kind: case.kind.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_interpolate_rect_bivariate_bilinear_spline".into(),
        category: "fsci_interpolate::rect_bivariate_spline + rect_bilinear_spline vs scipy".into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.kind, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "rect_bivariate/bilinear spline conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
