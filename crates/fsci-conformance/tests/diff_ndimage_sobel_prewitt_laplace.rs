#![forbid(unsafe_code)]
//! Live scipy parity for fsci_ndimage::{sobel, prewitt, laplace}.
//!
//! Resolves [frankenscipy-0f3dx]. Compares fsci's edge-detection
//! filters against scipy.ndimage.{sobel, prewitt, laplace} on a
//! 6×6 test image across all four boundary modes (reflect, constant,
//! nearest, wrap). For sobel/prewitt, both axes (0 and 1) are
//! exercised.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{BoundaryMode, NdArray, laplace, prewitt, sobel};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-10;
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    /// "sobel" | "prewitt" | "laplace"
    func: String,
    rows: usize,
    cols: usize,
    data: Vec<f64>,
    /// Axis (ignored for laplace)
    axis: usize,
    /// "reflect" | "constant" | "nearest" | "wrap"
    mode: String,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<CasePoint>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OraclePoint {
    case_id: String,
    out: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OraclePoint>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
    max_abs_diff: f64,
    max_rel_diff: f64,
    pass: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create sobel diff dir");
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

fn mode_from_str(s: &str) -> BoundaryMode {
    match s {
        "reflect" => BoundaryMode::Reflect,
        "constant" => BoundaryMode::Constant,
        "nearest" => BoundaryMode::Nearest,
        "wrap" => BoundaryMode::Wrap,
        _ => panic!("unknown mode {s}"),
    }
}

fn build_query() -> OracleQuery {
    // 6x6 test image with a clear gradient pattern
    let rows = 6;
    let cols = 6;
    let data: Vec<f64> = (0..rows * cols)
        .map(|i| (i as f64).sin() * 10.0 + i as f64)
        .collect();

    let modes = ["reflect", "constant", "nearest", "wrap"];
    let mut pts = Vec::new();

    for mode in &modes {
        // sobel along both axes
        for axis in [0_usize, 1] {
            pts.push(CasePoint {
                case_id: format!("sobel_axis{axis}_{mode}"),
                func: "sobel".into(),
                rows,
                cols,
                data: data.clone(),
                axis,
                mode: (*mode).into(),
            });
        }
        // prewitt along both axes
        for axis in [0_usize, 1] {
            pts.push(CasePoint {
                case_id: format!("prewitt_axis{axis}_{mode}"),
                func: "prewitt".into(),
                rows,
                cols,
                data: data.clone(),
                axis,
                mode: (*mode).into(),
            });
        }
        // laplace
        pts.push(CasePoint {
            case_id: format!("laplace_{mode}"),
            func: "laplace".into(),
            rows,
            cols,
            data: data.clone(),
            axis: 0,
            mode: (*mode).into(),
        });
    }

    OracleQuery { points: pts }
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
import numpy as np
import scipy.ndimage as ndi

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    cid = c["case_id"]
    try:
        arr = np.array(c["data"], dtype=float).reshape(c["rows"], c["cols"])
        mode = c["mode"]
        if c["func"] == "sobel":
            r = ndi.sobel(arr, axis=int(c["axis"]), mode=mode, cval=0.0)
        elif c["func"] == "prewitt":
            r = ndi.prewitt(arr, axis=int(c["axis"]), mode=mode, cval=0.0)
        elif c["func"] == "laplace":
            r = ndi.laplace(arr, mode=mode, cval=0.0)
        else:
            r = None
        if r is None or not np.all(np.isfinite(r)):
            out.append({"case_id": cid, "out": None})
        else:
            out.append({"case_id": cid, "out": [float(v) for v in r.flatten()]})
    except Exception:
        out.append({"case_id": cid, "out": None})

print(json.dumps({"points": out}))
"#;
    let query_json = serde_json::to_string(q).expect("serialize");
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
                "python3 spawn failed: {e}"
            );
            eprintln!("skipping sobel oracle: python3 unavailable ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping sobel oracle: stdin write failed");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "oracle failed: {stderr}"
        );
        eprintln!("skipping sobel oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

#[test]
fn diff_ndimage_sobel_prewitt_laplace() {
    let query = build_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    for (case, o) in query.points.iter().zip(oracle.points.iter()) {
        assert_eq!(case.case_id, o.case_id);
        let Some(expected) = o.out.as_ref() else {
            continue;
        };

        let arr =
            NdArray::new(case.data.clone(), vec![case.rows, case.cols]).expect("NdArray build");
        let mode = mode_from_str(&case.mode);
        let result = match case.func.as_str() {
            "sobel" => sobel(&arr, case.axis, mode, 0.0),
            "prewitt" => prewitt(&arr, case.axis, mode, 0.0),
            "laplace" => laplace(&arr, mode, 0.0),
            other => panic!("unknown func {other}"),
        };
        let actual = match result {
            Ok(a) => a,
            Err(e) => {
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    func: case.func.clone(),
                    max_abs_diff: f64::INFINITY,
                    max_rel_diff: f64::INFINITY,
                    pass: false,
                    note: format!("filter error: {e:?}"),
                });
                continue;
            }
        };

        if actual.data.len() != expected.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                max_abs_diff: f64::INFINITY,
                max_rel_diff: f64::INFINITY,
                pass: false,
                note: format!(
                    "length mismatch: fsci={} scipy={}",
                    actual.data.len(),
                    expected.len()
                ),
            });
            continue;
        }

        let mut max_abs = 0.0_f64;
        let mut max_rel = 0.0_f64;
        for (a, e) in actual.data.iter().zip(expected.iter()) {
            let abs_d = (a - e).abs();
            let denom = e.abs().max(1.0e-300);
            max_abs = max_abs.max(abs_d);
            max_rel = max_rel.max(abs_d / denom);
        }
        let pass = max_rel <= REL_TOL || max_abs <= ABS_TOL;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            max_abs_diff: max_abs,
            max_rel_diff: max_rel,
            pass,
            note: String::new(),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_ndimage_sobel_prewitt_laplace".into(),
        category: "fsci_ndimage::{sobel, prewitt, laplace} vs scipy.ndimage".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "edge filter mismatch: {} ({}) max_rel={} max_abs={} note={}",
                d.case_id, d.func, d.max_rel_diff, d.max_abs_diff, d.note
            );
        }
    }

    assert!(
        all_pass,
        "sobel/prewitt/laplace parity failed: {} cases",
        diffs.len()
    );
}
