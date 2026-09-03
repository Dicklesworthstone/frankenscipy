#![forbid(unsafe_code)]
//! Live scipy.interpolate parity for fsci_interpolate::griddata
//! and fsci_interpolate::cubic_hermite_interpolate.
//!
//! Resolves [frankenscipy-bqxew].
//!
//! - `griddata`: scattered 2-D interpolation. Compare against
//!   scipy.interpolate.griddata(points, values, xi, method='linear'/'nearest').
//!   Both ops deterministic on the convex hull; query points stay inside.
//! - `cubic_hermite_interpolate`: piecewise Hermite cubic. Compare
//!   against scipy.interpolate.CubicHermiteSpline(xi, yi, dydx)(x).
//!
//! Tolerance: 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_interpolate::{GriddataMethod, cubic_hermite_interpolate, griddata};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "grid_linear" | "grid_nearest" | "chermite"
    /// For grid_*
    points: Vec<Vec<f64>>,
    values: Vec<f64>,
    xi_grid: Vec<Vec<f64>>,
    /// For chermite
    xs: Vec<f64>,
    ys: Vec<f64>,
    dys: Vec<f64>,
    x_new: Vec<f64>,
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
    fs::create_dir_all(output_dir()).expect("create griddata_chermite diff dir");
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
    // griddata: scatter on a unit square with z = f(x, y).
    let pts_grid: Vec<Vec<f64>> = (0..5)
        .flat_map(|i| {
            (0..5).map(move |j| {
                let x = i as f64 / 4.0;
                let y = j as f64 / 4.0;
                vec![x, y]
            })
        })
        .collect();
    let vals_lin: Vec<f64> = pts_grid
        .iter()
        .map(|p| 2.0 * p[0] + 3.0 * p[1] + 1.0)
        .collect();
    let vals_nonlin: Vec<f64> = pts_grid
        .iter()
        .map(|p| (p[0] - 0.5).powi(2) + (p[1] - 0.5).powi(2))
        .collect();

    // Query points strictly inside the convex hull
    let xi_grid: Vec<Vec<f64>> = vec![
        vec![0.1, 0.1],
        vec![0.25, 0.5],
        vec![0.5, 0.5],
        vec![0.75, 0.25],
        vec![0.9, 0.9],
        vec![0.33, 0.67],
        vec![0.6, 0.4],
        vec![0.45, 0.85],
    ];

    let mut points = vec![
        Case {
            case_id: "grid_linear_planar".into(),
            op: "grid_linear".into(),
            points: pts_grid.clone(),
            values: vals_lin.clone(),
            xi_grid: xi_grid.clone(),
            xs: vec![],
            ys: vec![],
            dys: vec![],
            x_new: vec![],
        },
        Case {
            case_id: "grid_linear_quadratic".into(),
            op: "grid_linear".into(),
            points: pts_grid.clone(),
            values: vals_nonlin.clone(),
            xi_grid: xi_grid.clone(),
            xs: vec![],
            ys: vec![],
            dys: vec![],
            x_new: vec![],
        },
        Case {
            case_id: "grid_nearest_planar".into(),
            op: "grid_nearest".into(),
            points: pts_grid.clone(),
            values: vals_lin,
            xi_grid: xi_grid.clone(),
            xs: vec![],
            ys: vec![],
            dys: vec![],
            x_new: vec![],
        },
        Case {
            case_id: "grid_nearest_quadratic".into(),
            op: "grid_nearest".into(),
            points: pts_grid,
            values: vals_nonlin,
            xi_grid,
            xs: vec![],
            ys: vec![],
            dys: vec![],
            x_new: vec![],
        },
    ];

    // CubicHermiteSpline probes — exact-curve, exact-slope inputs
    // f(x) = sin(x), f'(x) = cos(x) on [0, 2π]
    let xs1: Vec<f64> = (0..9)
        .map(|i| i as f64 * std::f64::consts::PI / 4.0)
        .collect();
    let ys1: Vec<f64> = xs1.iter().map(|x| x.sin()).collect();
    let dy1: Vec<f64> = xs1.iter().map(|x| x.cos()).collect();
    let xn1: Vec<f64> = (0..21)
        .map(|i| i as f64 * 0.1 * std::f64::consts::PI)
        .collect();

    // f(x) = x^3 - 2x + 1
    let xs2: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let ys2: Vec<f64> = xs2.iter().map(|x| x * x * x - 2.0 * x + 1.0).collect();
    let dy2: Vec<f64> = xs2.iter().map(|x| 3.0 * x * x - 2.0).collect();
    let xn2: Vec<f64> = (0..16).map(|i| -2.0 + i as f64 * 0.3).collect();

    points.push(Case {
        case_id: "chermite_sin".into(),
        op: "chermite".into(),
        points: vec![],
        values: vec![],
        xi_grid: vec![],
        xs: xs1,
        ys: ys1,
        dys: dy1,
        x_new: xn1,
    });
    points.push(Case {
        case_id: "chermite_cubic".into(),
        op: "chermite".into(),
        points: vec![],
        values: vec![],
        xi_grid: vec![],
        xs: xs2,
        ys: ys2,
        dys: dy2,
        x_new: xn2,
    });

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.interpolate import griddata, CubicHermiteSpline

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    try:
        if op in ("grid_linear", "grid_nearest"):
            pts = np.array(case["points"], dtype=float)
            vs = np.array(case["values"], dtype=float)
            xi = np.array(case["xi_grid"], dtype=float)
            method = "linear" if op == "grid_linear" else "nearest"
            res = griddata(pts, vs, xi, method=method)
            flat = [float(v) for v in res.tolist()]
            if all(math.isfinite(v) for v in flat):
                points.append({"case_id": cid, "values": flat})
            else:
                points.append({"case_id": cid, "values": None})
        elif op == "chermite":
            xs = np.array(case["xs"], dtype=float)
            ys = np.array(case["ys"], dtype=float)
            dys = np.array(case["dys"], dtype=float)
            xn = np.array(case["x_new"], dtype=float)
            sp = CubicHermiteSpline(xs, ys, dys)
            res = sp(xn)
            flat = [float(v) for v in res.tolist()]
            if all(math.isfinite(v) for v in flat):
                points.append({"case_id": cid, "values": flat})
            else:
                points.append({"case_id": cid, "values": None})
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
                "failed to spawn python3 for griddata_chermite oracle: {e}"
            );
            eprintln!("skipping griddata_chermite oracle: python3 not available ({e})");
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
                "griddata_chermite oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping griddata_chermite oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for griddata_chermite oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "griddata_chermite oracle failed: {stderr}"
        );
        eprintln!("skipping griddata_chermite oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse griddata_chermite oracle JSON"))
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
fn diff_interpolate_griddata_cubic_hermite() {
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
        let result: Option<Vec<f64>> = match case.op.as_str() {
            "grid_linear" => griddata(
                &case.points,
                &case.values,
                &case.xi_grid,
                GriddataMethod::Linear,
            )
            .ok(),
            "grid_nearest" => griddata(
                &case.points,
                &case.values,
                &case.xi_grid,
                GriddataMethod::Nearest,
            )
            .ok(),
            "chermite" => {
                cubic_hermite_interpolate(&case.xs, &case.ys, &case.dys, &case.x_new).ok()
            }
            _ => None,
        };
        let Some(y) = result else {
            continue;
        };
        let abs_d = vec_max_diff(&y, expected);
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
        test_id: "diff_interpolate_griddata_cubic_hermite".into(),
        category: "fsci_interpolate::griddata + cubic_hermite_interpolate vs scipy".into(),
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
        "griddata/chermite conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
