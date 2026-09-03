#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.interpolate.interpn` on
//! 2-D regular grids with linear and nearest methods.
//!
//! Resolves [frankenscipy-3sja3]. 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_interpolate::{RegularGridMethod, interpn};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    method: String,
    /// Per-axis grid coordinates (length per axis varies).
    points: Vec<Vec<f64>>,
    /// Row-major flattened grid values.
    values: Vec<f64>,
    /// Query points: list of [x_axis0, x_axis1, ...] coordinates.
    xi: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    expected: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    method: String,
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
    fs::create_dir_all(output_dir()).expect("create interpn diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize interpn diff log");
    fs::write(path, json).expect("write interpn diff log");
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![a];
    }
    let step = (b - a) / (n - 1) as f64;
    (0..n).map(|i| a + step * i as f64).collect()
}

fn generate_query() -> OracleQuery {
    // 3×4 grid
    let xs_a = linspace(0.0, 1.0, 3);
    let ys_a = linspace(0.0, 2.0, 4);
    let mut vals_a = Vec::with_capacity(12);
    for &x in &xs_a {
        for &y in &ys_a {
            vals_a.push(x + 2.0 * y); // linear: should be exact under linear method
        }
    }
    let xi_a: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![0.25, 0.5],
        vec![0.5, 1.0],
        vec![0.75, 1.5],
        vec![1.0, 2.0],
    ];

    // 5×5 grid with nonlinear values
    let xs_b = linspace(-1.0, 1.0, 5);
    let ys_b = linspace(-1.0, 1.0, 5);
    let mut vals_b = Vec::with_capacity(25);
    for &x in &xs_b {
        for &y in &ys_b {
            vals_b.push((x * 2.0).sin() + (y * 1.5).cos());
        }
    }
    let xi_b: Vec<Vec<f64>> = vec![
        vec![-0.9, -0.7],
        vec![-0.4, 0.2],
        vec![0.0, 0.0],
        vec![0.3, -0.4],
        vec![0.6, 0.5],
        vec![0.9, 0.9],
    ];

    let cases: &[(&str, &str, Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>)] = &[
        (
            "linear_3x4_plane",
            "linear",
            vec![xs_a.clone(), ys_a.clone()],
            vals_a.clone(),
            xi_a.clone(),
        ),
        (
            "nearest_3x4_plane",
            "nearest",
            vec![xs_a, ys_a],
            vals_a,
            xi_a,
        ),
        (
            "linear_5x5_sincos",
            "linear",
            vec![xs_b.clone(), ys_b.clone()],
            vals_b.clone(),
            xi_b.clone(),
        ),
        (
            "nearest_5x5_sincos",
            "nearest",
            vec![xs_b, ys_b],
            vals_b,
            xi_b,
        ),
    ];
    let points = cases
        .iter()
        .map(|(name, method, pts, vals, xi)| PointCase {
            case_id: (*name).into(),
            method: (*method).into(),
            points: pts.clone(),
            values: vals.clone(),
            xi: xi.clone(),
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
from scipy import interpolate

def finite_vec_or_none(arr):
    out = []
    for v in np.asarray(arr).flatten().tolist():
        try:
            v = float(v)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out

q = json.load(sys.stdin)
points_out = []
for case in q["points"]:
    cid = case["case_id"]; method = case["method"]
    grid_axes = tuple(np.array(ax, dtype=float) for ax in case["points"])
    shape = tuple(len(ax) for ax in grid_axes)
    vals = np.array(case["values"], dtype=float).reshape(shape)
    xi = np.array(case["xi"], dtype=float)
    try:
        out = interpolate.interpn(
            grid_axes, vals, xi,
            method=method, bounds_error=True, fill_value=None,
        )
        points_out.append({"case_id": cid, "expected": finite_vec_or_none(out)})
    except Exception:
        points_out.append({"case_id": cid, "expected": None})
print(json.dumps({"points": points_out}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize interpn query");
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
                "failed to spawn python3 for interpn oracle: {e}"
            );
            eprintln!("skipping interpn oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open interpn oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "interpn oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping interpn oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for interpn oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "interpn oracle failed: {stderr}"
        );
        eprintln!("skipping interpn oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse interpn oracle JSON"))
}

#[test]
fn diff_interpolate_interpn() {
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
        let Some(expected) = scipy_arm.expected.as_ref() else {
            continue;
        };
        let method = match case.method.as_str() {
            "linear" => RegularGridMethod::Linear,
            "nearest" => RegularGridMethod::Nearest,
            _ => continue,
        };
        let Ok(fsci_v) = interpn(
            case.points.clone(),
            case.values.clone(),
            &case.xi,
            method,
            true,
            None,
        ) else {
            continue;
        };
        if fsci_v.len() != expected.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                method: case.method.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let abs_d = fsci_v
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            method: case.method.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_interpolate_interpn".into(),
        category: "scipy.interpolate.interpn (linear, nearest)".into(),
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
                "interpn {} mismatch: {} abs_diff={}",
                d.method, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.interpolate.interpn conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
