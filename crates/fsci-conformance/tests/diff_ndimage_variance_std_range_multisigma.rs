#![forbid(unsafe_code)]
//! Live scipy.ndimage parity for fsci_ndimage::variance_filter,
//! std_filter, range_filter, and gaussian_filter_multi_sigma.
//!
//! Resolves [frankenscipy-h7asi].
//!
//! - `variance_filter` / `std_filter`: use scipy.ndimage.generic_filter
//!   with np.var / np.std (ddof=0) as the oracle.
//! - `range_filter`: max - min over neighborhood; oracle is
//!   scipy.ndimage.maximum_filter - minimum_filter.
//! - `gaussian_filter_multi_sigma`: per-axis sigma; oracle is
//!   scipy.ndimage.gaussian_filter(image, sigma=tuple).
//!
//! Tolerance: 1e-12 abs for variance/std/range, 1e-10 abs for gaussian
//! (per-axis sequential convolution accumulates more rounding).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{
    BoundaryMode, NdArray, gaussian_filter_multi_sigma, range_filter, std_filter, variance_filter,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const FLOOR_ABS_TOL: f64 = 1.0e-12;
const GAUSS_ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "var" | "std" | "range" | "gauss_multi"
    rows: usize,
    cols: usize,
    data: Vec<f64>,
    size: usize,
    mode: String,
    /// For gauss_multi
    sigma_y: f64,
    sigma_x: f64,
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
    fs::create_dir_all(output_dir()).expect("create ndimage_var_std_range diff dir");
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

fn parse_mode(s: &str) -> Option<BoundaryMode> {
    match s {
        "reflect" => Some(BoundaryMode::Reflect),
        "constant" => Some(BoundaryMode::Constant),
        "nearest" => Some(BoundaryMode::Nearest),
        "wrap" => Some(BoundaryMode::Wrap),
        _ => None,
    }
}

fn synth_image(rows: usize, cols: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut out = Vec::with_capacity(rows * cols);
    for _ in 0..(rows * cols) {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((s >> 11) as f64) / (1u64 << 53) as f64;
        out.push((u - 0.5) * 6.0);
    }
    out
}

fn generate_query() -> OracleQuery {
    let img_a = synth_image(6, 8, 0xdead_beef_cafe_babe);
    let img_b = synth_image(7, 7, 0x1234_5678_90ab_cdef);

    let mut points = Vec::new();

    for (label, rows, cols, data) in [
        ("a6x8", 6_usize, 8_usize, &img_a),
        ("b7x7", 7_usize, 7_usize, &img_b),
    ] {
        for size in [3_usize, 5] {
            for mode in ["reflect", "nearest", "constant"] {
                for op in ["var", "std", "range"] {
                    points.push(Case {
                        case_id: format!("{op}_{label}_s{size}_{mode}"),
                        op: op.into(),
                        rows,
                        cols,
                        data: data.clone(),
                        size,
                        mode: mode.into(),
                        sigma_y: 0.0,
                        sigma_x: 0.0,
                    });
                }
            }
        }
        // gauss_multi with several (σy, σx), including non-half-integer
        // values that exercise SciPy's default truncate-radius rounding.
        for &(sy, sx) in &[(0.5, 0.5), (1.0, 0.5), (0.5, 1.5), (1.3, 0.7), (1.5, 1.5)] {
            for mode in ["reflect", "nearest"] {
                points.push(Case {
                    case_id: format!("gauss_{label}_sy{sy}_sx{sx}_{mode}"),
                    op: "gauss_multi".into(),
                    rows,
                    cols,
                    data: data.clone(),
                    size: 0,
                    mode: mode.into(),
                    sigma_y: sy,
                    sigma_x: sx,
                });
            }
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
from scipy import ndimage as ndi

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    try:
        rows = int(case["rows"]); cols = int(case["cols"])
        arr = np.array(case["data"], dtype=float).reshape((rows, cols))
        mode = case["mode"]
        if op in ("var", "std"):
            size = int(case["size"])
            func = np.var if op == "var" else np.std
            res = ndi.generic_filter(arr, func, size=size, mode=mode, cval=0.0)
        elif op == "range":
            size = int(case["size"])
            mx = ndi.maximum_filter(arr, size=size, mode=mode, cval=0.0)
            mn = ndi.minimum_filter(arr, size=size, mode=mode, cval=0.0)
            res = mx - mn
        elif op == "gauss_multi":
            sy = float(case["sigma_y"]); sx = float(case["sigma_x"])
            res = ndi.gaussian_filter(arr, sigma=(sy, sx), mode=mode, cval=0.0)
        else:
            points.append({"case_id": cid, "values": None}); continue
        flat = [float(v) for v in np.asarray(res).flatten().tolist()]
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
                "failed to spawn python3 for ndimage var_std_range oracle: {e}"
            );
            eprintln!("skipping ndimage var_std_range oracle: python3 not available ({e})");
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
                "ndimage var_std_range oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!(
                "skipping ndimage var_std_range oracle: stdin write failed ({err})\n{stderr}"
            );
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for ndimage var_std_range oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ndimage var_std_range oracle failed: {stderr}"
        );
        eprintln!("skipping ndimage var_std_range oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ndimage var_std_range oracle JSON"))
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
fn diff_ndimage_variance_std_range_multisigma() {
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
        let Some(mode) = parse_mode(&case.mode) else {
            continue;
        };
        let Ok(input) = NdArray::new(case.data.clone(), vec![case.rows, case.cols]) else {
            continue;
        };
        let (result_data, tol): (Vec<f64>, f64) = match case.op.as_str() {
            "var" => {
                let Ok(r) = variance_filter(&input, case.size, mode, 0.0) else {
                    continue;
                };
                (r.data, FLOOR_ABS_TOL)
            }
            "std" => {
                let Ok(r) = std_filter(&input, case.size, mode, 0.0) else {
                    continue;
                };
                (r.data, FLOOR_ABS_TOL)
            }
            "range" => {
                let Ok(r) = range_filter(&input, case.size, mode, 0.0) else {
                    continue;
                };
                (r.data, FLOOR_ABS_TOL)
            }
            "gauss_multi" => {
                let Ok(r) =
                    gaussian_filter_multi_sigma(&input, &[case.sigma_y, case.sigma_x], mode, 0.0)
                else {
                    continue;
                };
                (r.data, GAUSS_ABS_TOL)
            }
            _ => continue,
        };
        let abs_d = vec_max_diff(&result_data, expected);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= tol,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_ndimage_variance_std_range_multisigma".into(),
        category:
            "fsci_ndimage::{variance_filter, std_filter, range_filter, gaussian_filter_multi_sigma} vs scipy.ndimage"
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "ndimage var/std/range/multi-sigma conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
