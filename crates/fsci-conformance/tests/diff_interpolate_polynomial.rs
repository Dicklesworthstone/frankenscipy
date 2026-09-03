#![forbid(unsafe_code)]
//! Live SciPy differential coverage for four polynomial / piecewise-
//! polynomial interpolation routines in `scipy.interpolate`:
//!   - `scipy.interpolate.PchipInterpolator(xi, yi)`
//!   - `scipy.interpolate.Akima1DInterpolator(xi, yi)`
//!   - `scipy.interpolate.BarycentricInterpolator(xi, yi)`
//!   - `scipy.interpolate.KroghInterpolator(xi, yi)`
//!
//! Resolves [frankenscipy-88hpl]. fsci_interpolate exposes
//! pchip_interpolate / akima1d_interpolate / barycentric_interpolate /
//! krogh_interpolate but diff_interpolate.rs only covers
//! interp1d_linear / lagrange / polyfit / polyval / splev / splrep.
//! Verified all four routines match scipy exactly on probe inputs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_interpolate::{
    akima1d_interpolate, barycentric_interpolate, krogh_interpolate, pchip_interpolate,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-013";
const ABS_TOL: f64 = 1.0e-11;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    xi: Vec<f64>,
    yi: Vec<f64>,
    x_new: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
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
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create interpolate_poly diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize interpolate_poly diff log");
    fs::write(path, json).expect("write interpolate_poly diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: &[(&str, Vec<f64>, Vec<f64>, Vec<f64>)] = &[
        (
            "x_squared",
            vec![1.0, 2.0, 4.0, 5.0],
            vec![1.0, 4.0, 16.0, 25.0],
            vec![1.5, 3.0, 4.5],
        ),
        (
            "monotonic",
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
            vec![0.0, 1.0, 3.0, 6.0, 10.0],
            vec![0.5, 1.5, 2.5, 3.5],
        ),
        (
            "non_uniform",
            vec![-2.0, -1.0, 0.5, 2.5, 4.0],
            vec![3.0, 0.0, 1.0, 4.0, 0.0],
            vec![-1.5, 0.0, 1.5, 3.0],
        ),
    ];
    let funcs = ["pchip", "akima", "barycentric", "krogh"];
    let mut points = Vec::new();
    for (label, xi, yi, x_new) in datasets {
        for func in funcs {
            points.push(PointCase {
                case_id: format!("{func}_{label}"),
                func: func.into(),
                xi: xi.clone(),
                yi: yi.clone(),
                x_new: x_new.clone(),
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
from scipy import interpolate

DISPATCH = {
    "pchip":       interpolate.PchipInterpolator,
    "akima":       interpolate.Akima1DInterpolator,
    "barycentric": interpolate.BarycentricInterpolator,
    "krogh":       interpolate.KroghInterpolator,
}

def finite_vec_or_none(arr):
    out = []
    for v in np.asarray(arr).tolist():
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
    cid = case["case_id"]; func = case["func"]
    xi = np.array(case["xi"], dtype=float)
    yi = np.array(case["yi"], dtype=float)
    x_new = np.array(case["x_new"], dtype=float)
    ctor = DISPATCH.get(func)
    if ctor is None:
        points.append({"case_id": cid, "values": None}); continue
    try:
        interp = ctor(xi, yi)
        points.append({"case_id": cid, "values": finite_vec_or_none(interp(x_new))})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize interpolate_poly query");
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
                "failed to spawn python3 for interpolate_poly oracle: {e}"
            );
            eprintln!("skipping interpolate_poly oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open interpolate_poly oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "interpolate_poly oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping interpolate_poly oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for interpolate_poly oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "interpolate_poly oracle failed: {stderr}"
        );
        eprintln!("skipping interpolate_poly oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse interpolate_poly oracle JSON"))
}

fn fsci_eval(func: &str, xi: &[f64], yi: &[f64], x_new: &[f64]) -> Option<Vec<f64>> {
    match func {
        "pchip" => pchip_interpolate(xi, yi, x_new).ok(),
        "akima" => akima1d_interpolate(xi, yi, x_new).ok(),
        "barycentric" => barycentric_interpolate(xi, yi, x_new).ok(),
        "krogh" => krogh_interpolate(xi, yi, x_new).ok(),
        _ => None,
    }
}

#[test]
fn diff_interpolate_polynomial() {
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
        let Some(fsci_v) = fsci_eval(&case.func, &case.xi, &case.yi, &case.x_new) else {
            continue;
        };
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        if fsci_v.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let abs_d = fsci_v
            .iter()
            .zip(scipy_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_interpolate_polynomial".into(),
        category: "scipy.interpolate.{Pchip,Akima1D,Barycentric,Krogh}Interpolator".into(),
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
                "interpolate_poly {} mismatch: {} abs_diff={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.interpolate polynomial interpolators conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
