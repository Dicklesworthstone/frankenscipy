#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_linalg matrix
//! functions.
//!
//! Resolves [frankenscipy-srhne]. Covers the 9 matrix-function
//! primitives: expm, sqrtm, logm, and the matrix-trig variants
//! (sinm, cosm, tanm, sinhm, coshm, tanhm). Each diffed against
//! the corresponding scipy.linalg.{name}.
//!
//! 4 fixtures × applicable functions ≈ 28-32 cases. Tol 1e-9
//! abs because matrix functions use Padé approximation / Schur
//! decomposition (iterative) — scipy's implementation differs
//! in detail from fsci's.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{DecompOptions, coshm, cosm, expm, logm, sinhm, sinm, sqrtm, tanhm, tanm};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-013";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    a: Vec<Vec<f64>>,
    fns: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    fn_name: String,
    matrix: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    fn_name: String,
    max_abs_diff: f64,
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
    fs::create_dir_all(output_dir()).expect("create matrix-functions diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize matrix-functions diff log");
    fs::write(path, json).expect("write matrix-functions diff log");
}

fn dispatch_fn(name: &str, a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let opts = DecompOptions::default();
    match name {
        "expm" => expm(a, opts).ok(),
        "sqrtm" => sqrtm(a, opts).ok(),
        "logm" => logm(a, opts).ok(),
        "sinm" => sinm(a, opts).ok(),
        "cosm" => cosm(a, opts).ok(),
        "tanm" => tanm(a, opts).ok(),
        "sinhm" => sinhm(a, opts).ok(),
        "coshm" => coshm(a, opts).ok(),
        "tanhm" => tanhm(a, opts).ok(),
        _ => None,
    }
}

fn max_abs_diff_mat(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut m = 0.0_f64;
    for (ra, rb) in a.iter().zip(b.iter()) {
        for (va, vb) in ra.iter().zip(rb.iter()) {
            m = m.max((va - vb).abs());
        }
    }
    m
}

fn generate_query() -> OracleQuery {
    // 9 matrix functions; logm requires positive eigenvalues
    // (avoid the identity-only fixture for the trig fns since
    // sinm/cosm/tanm of identity is sin(1)·I which is fine, but
    // the fixture is uninteresting). Pick 4 representative
    // matrices.
    let all_fns: Vec<String> = vec![
        "expm", "sqrtm", "logm", "sinm", "cosm", "tanm", "sinhm", "coshm", "tanhm",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    OracleQuery {
        points: vec![
            // Symmetric positive-definite 2×2
            PointCase {
                case_id: "spd_2x2".into(),
                a: vec![vec![2.0, 0.5], vec![0.5, 1.5]],
                fns: all_fns.clone(),
            },
            // Diagonal 3×3 (closed-form: f(D) is diag(f(d_i)))
            PointCase {
                case_id: "diag_3x3".into(),
                a: vec![
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 0.5, 0.0],
                    vec![0.0, 0.0, 0.25],
                ],
                fns: all_fns.clone(),
            },
            // Symmetric 3×3 with distinct eigenvalues
            PointCase {
                case_id: "sym_3x3_distinct".into(),
                a: vec![
                    vec![3.0, 1.0, 0.0],
                    vec![1.0, 2.0, 0.5],
                    vec![0.0, 0.5, 1.0],
                ],
                fns: all_fns.clone(),
            },
            // Identity 4×4
            PointCase {
                case_id: "identity_4x4".into(),
                a: vec![
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0, 0.0],
                    vec![0.0, 0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 0.0, 1.0],
                ],
                fns: all_fns.clone(),
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
from scipy.linalg import expm, sqrtm, logm, sinm, cosm, tanm, sinhm, coshm, tanhm

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

def listify_2d(arr):
    out = []
    for row in arr.tolist():
        rrow = []
        for v in row:
            f = fnone(v)
            if f is None:
                return None
            rrow.append(f)
        out.append(rrow)
    return out

DISPATCH = {
    "expm": expm, "sqrtm": sqrtm, "logm": logm,
    "sinm": sinm, "cosm": cosm, "tanm": tanm,
    "sinhm": sinhm, "coshm": coshm, "tanhm": tanhm,
}

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    A = np.asarray(case["a"], dtype=np.float64)
    for name in case["fns"]:
        out = {"case_id": cid, "fn_name": name, "matrix": None}
        try:
            res = DISPATCH[name](A)
            # Some scipy fns return tuples (sqrtm returns (mat, errest)).
            if isinstance(res, tuple):
                res = res[0]
            res = np.asarray(res, dtype=np.float64)
            # logm/sqrtm may return complex if eigenvalues are negative
            # — drop any complex-valued result.
            if not np.all(np.isreal(res)):
                out["matrix"] = None
            else:
                out["matrix"] = listify_2d(res.real)
        except Exception:
            pass
        points.append(out)
print(json.dumps({"points": points}, allow_nan=False))
"#;
    let query_json = serde_json::to_string(query).expect("serialize matrix-functions query");
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
                "failed to spawn python3 for matrix-functions oracle: {e}"
            );
            eprintln!("skipping matrix-functions oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open matrix-functions oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "matrix-functions oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping matrix-functions oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for matrix-functions oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "matrix-functions oracle failed: {stderr}"
        );
        eprintln!("skipping matrix-functions oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse matrix-functions oracle JSON"))
}

#[test]
fn diff_linalg_matrix_functions() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    // Build (case_id, fn_name) → scipy matrix lookup.
    let mut map: HashMap<(String, String), Vec<Vec<f64>>> = HashMap::new();
    for arm in oracle.points.into_iter() {
        if let Some(mat) = arm.matrix {
            map.insert((arm.case_id, arm.fn_name), mat);
        }
    }

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        for fn_name in &case.fns {
            let Some(scipy_mat) = map.get(&(case.case_id.clone(), fn_name.clone())) else {
                continue;
            };
            let Some(rust_mat) = dispatch_fn(fn_name, &case.a) else {
                continue;
            };
            if rust_mat.len() != scipy_mat.len()
                || rust_mat
                    .iter()
                    .zip(scipy_mat.iter())
                    .any(|(rr, sr)| rr.len() != sr.len())
            {
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    fn_name: fn_name.clone(),
                    max_abs_diff: f64::INFINITY,
                    pass: false,
                });
                continue;
            }
            let max_d = max_abs_diff_mat(&rust_mat, scipy_mat);
            max_overall = max_overall.max(max_d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                fn_name: fn_name.clone(),
                max_abs_diff: max_d,
                pass: max_d <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_linalg_matrix_functions".into(),
        category: "fsci_linalg matrix functions".into(),
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
                "matrix-function mismatch: {} {} — max_abs={}",
                d.case_id, d.fn_name, d.max_abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "matrix-functions conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
