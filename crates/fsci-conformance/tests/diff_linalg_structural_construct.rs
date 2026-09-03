#![forbid(unsafe_code)]
//! Live numpy/SciPy differential coverage for fsci_linalg structural
//! constructors not covered by diff_linalg_structured_matrices or
//! misc_deterministic: `tri`, `tril`, `triu`, `vander`, `leslie`,
//! `eye_k`.
//!
//! Resolves [frankenscipy-3bv3e]. 1e-12 abs (integer / fractional data).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{eye_k, leslie, tri, tril, triu, vander};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    /// (n, m, k) for tri/eye_k.
    n: usize,
    m: usize,
    k: i64,
    /// Source matrix (row-major flat). For tril/triu only.
    a: Vec<f64>,
    a_rows: usize,
    a_cols: usize,
    /// For vander: x values, n (output cols), increasing flag.
    x: Vec<f64>,
    increasing: bool,
    /// For leslie: f (fertility) and s (survival).
    f_vals: Vec<f64>,
    s_vals: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    rows: Option<usize>,
    cols: Option<usize>,
    dense: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create structural_construct diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize structural_construct diff log");
    fs::write(path, json).expect("write structural_construct diff log");
}

fn flatten(m: &[Vec<f64>]) -> Vec<f64> {
    m.iter().flat_map(|row| row.iter().copied()).collect()
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // tri (boolean lower-triangle of ones)
    let tri_cases: &[(&str, usize, usize, i64)] = &[
        ("tri_4x4_k0", 4, 4, 0),
        ("tri_3x5_k1", 3, 5, 1),
        ("tri_5x3_kneg1", 5, 3, -1),
        ("tri_4x4_k2", 4, 4, 2),
    ];
    for (name, n, m, k) in tri_cases {
        points.push(PointCase {
            case_id: (*name).into(),
            op: "tri".into(),
            n: *n,
            m: *m,
            k: *k,
            a: vec![],
            a_rows: 0,
            a_cols: 0,
            x: vec![],
            increasing: false,
            f_vals: vec![],
            s_vals: vec![],
        });
    }

    // tril / triu (extract triangular part of a matrix)
    let mat_4x4: Vec<f64> = (1..=16).map(|i| i as f64).collect();
    let mat_3x5: Vec<f64> = (1..=15).map(|i| i as f64).collect();
    let mat_cases: &[(&str, &str, &[f64], usize, usize, i64)] = &[
        ("tril_4x4_k0", "tril", &mat_4x4, 4, 4, 0),
        ("tril_4x4_kneg1", "tril", &mat_4x4, 4, 4, -1),
        ("tril_3x5_k1", "tril", &mat_3x5, 3, 5, 1),
        ("triu_4x4_k0", "triu", &mat_4x4, 4, 4, 0),
        ("triu_4x4_k1", "triu", &mat_4x4, 4, 4, 1),
        ("triu_3x5_kneg1", "triu", &mat_3x5, 3, 5, -1),
    ];
    for (name, op, a, r, c, k) in mat_cases {
        points.push(PointCase {
            case_id: (*name).into(),
            op: (*op).into(),
            n: 0,
            m: 0,
            k: *k,
            a: a.to_vec(),
            a_rows: *r,
            a_cols: *c,
            x: vec![],
            increasing: false,
            f_vals: vec![],
            s_vals: vec![],
        });
    }

    // vander (x, n, increasing)
    let vander_cases: &[(&str, Vec<f64>, usize, bool)] = &[
        (
            "vander_default_n_decreasing",
            vec![1.0, 2.0, 3.0, 4.0],
            4,
            false,
        ),
        ("vander_n3_increasing", vec![1.0, 2.0, 3.0, 4.0], 3, true),
        ("vander_n5_decreasing", vec![0.5, 1.5, -1.0], 5, false),
        ("vander_n2_increasing", vec![2.0, 4.0, 6.0, 8.0], 2, true),
    ];
    for (name, x, n, inc) in vander_cases {
        points.push(PointCase {
            case_id: (*name).into(),
            op: "vander".into(),
            n: *n,
            m: 0,
            k: 0,
            a: vec![],
            a_rows: 0,
            a_cols: 0,
            x: x.clone(),
            increasing: *inc,
            f_vals: vec![],
            s_vals: vec![],
        });
    }

    // leslie
    let leslie_cases: &[(&str, Vec<f64>, Vec<f64>)] = &[
        ("leslie_3_class", vec![0.1, 2.0, 1.5], vec![0.2, 0.8]),
        (
            "leslie_4_class",
            vec![0.0, 1.5, 1.0, 0.5],
            vec![0.9, 0.7, 0.4],
        ),
    ];
    for (name, f_vals, s_vals) in leslie_cases {
        points.push(PointCase {
            case_id: (*name).into(),
            op: "leslie".into(),
            n: 0,
            m: 0,
            k: 0,
            a: vec![],
            a_rows: 0,
            a_cols: 0,
            x: vec![],
            increasing: false,
            f_vals: f_vals.clone(),
            s_vals: s_vals.clone(),
        });
    }

    // eye_k (rectangular shifted identity)
    let eye_cases: &[(&str, usize, usize, i64)] = &[
        ("eye_k_3x5_k1", 3, 5, 1),
        ("eye_k_5x3_kneg2", 5, 3, -2),
        ("eye_k_4x4_k0", 4, 4, 0),
        ("eye_k_3x3_k2", 3, 3, 2),
    ];
    for (name, n, m, k) in eye_cases {
        points.push(PointCase {
            case_id: (*name).into(),
            op: "eye_k".into(),
            n: *n,
            m: *m,
            k: *k,
            a: vec![],
            a_rows: 0,
            a_cols: 0,
            x: vec![],
            increasing: false,
            f_vals: vec![],
            s_vals: vec![],
        });
    }

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import linalg

def dense_or_none(arr):
    arr = np.asarray(arr, dtype=float)
    rows, cols = arr.shape
    flat = []
    for row in arr.tolist():
        for v in row:
            if not math.isfinite(float(v)):
                return rows, cols, None
            flat.append(float(v))
    return rows, cols, flat

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    try:
        if op == "tri":
            m = np.tri(int(case["n"]), int(case["m"]), k=int(case["k"]))
        elif op == "tril":
            src = np.array(case["a"], dtype=float).reshape(int(case["a_rows"]), int(case["a_cols"]))
            m = np.tril(src, k=int(case["k"]))
        elif op == "triu":
            src = np.array(case["a"], dtype=float).reshape(int(case["a_rows"]), int(case["a_cols"]))
            m = np.triu(src, k=int(case["k"]))
        elif op == "vander":
            x = np.array(case["x"], dtype=float)
            m = np.vander(x, N=int(case["n"]), increasing=bool(case["increasing"]))
        elif op == "leslie":
            f = np.array(case["f_vals"], dtype=float)
            s = np.array(case["s_vals"], dtype=float)
            m = linalg.leslie(f, s)
        elif op == "eye_k":
            m = np.eye(int(case["n"]), int(case["m"]), k=int(case["k"]))
        else:
            m = None
        if m is None:
            points.append({"case_id": cid, "rows": None, "cols": None, "dense": None})
        else:
            r, c, dn = dense_or_none(m)
            points.append({"case_id": cid, "rows": r, "cols": c, "dense": dn})
    except Exception:
        points.append({"case_id": cid, "rows": None, "cols": None, "dense": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize structural_construct query");
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
                "failed to spawn python3 for structural_construct oracle: {e}"
            );
            eprintln!("skipping structural_construct oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open structural_construct oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "structural_construct oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping structural_construct oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for structural_construct oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "structural_construct oracle failed: {stderr}"
        );
        eprintln!("skipping structural_construct oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse structural_construct oracle JSON"))
}

fn rows_of(a_flat: &[f64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|r| (0..cols).map(|c| a_flat[r * cols + c]).collect())
        .collect()
}

#[test]
fn diff_linalg_structural_construct() {
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
        let Some(expected) = scipy_arm.dense.as_ref() else {
            continue;
        };
        let (Some(rows), Some(cols)) = (scipy_arm.rows, scipy_arm.cols) else {
            continue;
        };
        let fsci_mat: Vec<Vec<f64>> = match case.op.as_str() {
            "tri" => tri(case.n, case.m, case.k),
            "tril" => {
                let src = rows_of(&case.a, case.a_rows, case.a_cols);
                tril(&src, case.k)
            }
            "triu" => {
                let src = rows_of(&case.a, case.a_rows, case.a_cols);
                triu(&src, case.k)
            }
            "vander" => vander(&case.x, Some(case.n), case.increasing),
            "leslie" => match leslie(&case.f_vals, &case.s_vals) {
                Ok(m) => m,
                Err(_) => continue,
            },
            "eye_k" => eye_k(case.n, case.m, case.k),
            _ => continue,
        };
        let fsci_rows = fsci_mat.len();
        let fsci_cols = fsci_mat.first().map_or(0, |r| r.len());
        if fsci_rows != rows || fsci_cols != cols {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                op: case.op.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let fsci_flat = flatten(&fsci_mat);
        let abs_d = fsci_flat
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
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
        test_id: "diff_linalg_structural_construct".into(),
        category: "numpy/scipy.linalg tri+tril+triu+vander+leslie+eye_k".into(),
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
        "structural_construct conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
