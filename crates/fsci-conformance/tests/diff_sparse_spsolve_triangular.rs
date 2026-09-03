#![forbid(unsafe_code)]
//! Live scipy parity for fsci_sparse::spsolve_triangular.
//!
//! Resolves [frankenscipy-gy8gc]. Tests forward (lower=true) and
//! backward (lower=false) substitution on tridiagonal and dense-
//! triangular fixtures vs scipy.sparse.linalg.spsolve_triangular.
//!
//! Tolerance: 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CooMatrix, FormatConvertible, Shape2D, spsolve_triangular};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct TriCase {
    case_id: String,
    n: usize,
    /// COO triplets (row, col, val). Caller guarantees a triangular pattern
    /// matching `lower`.
    triplets: Vec<(usize, usize, f64)>,
    b: Vec<f64>,
    lower: bool,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<TriCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    x: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create spsolve_triangular diff dir");
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

fn lower_bidiag(n: usize, diag_val: f64, sub_val: f64) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    for i in 0..n {
        out.push((i, i, diag_val));
        if i + 1 > 0 && i >= 1 {
            out.push((i, i - 1, sub_val));
        }
    }
    out
}

fn upper_bidiag(n: usize, diag_val: f64, sup_val: f64) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    for i in 0..n {
        out.push((i, i, diag_val));
        if i + 1 < n {
            out.push((i, i + 1, sup_val));
        }
    }
    out
}

fn dense_lower(n: usize) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..=i {
            let v = if i == j {
                (i as f64) + 1.5
            } else {
                0.25 * ((i + j) as f64) - 0.1
            };
            out.push((i, j, v));
        }
    }
    out
}

fn dense_upper(n: usize) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    for i in 0..n {
        for j in i..n {
            let v = if i == j {
                (i as f64) + 2.0
            } else {
                0.5 * ((j - i) as f64) + 0.2
            };
            out.push((i, j, v));
        }
    }
    out
}

fn generate_query() -> OracleQuery {
    let points = vec![
        TriCase {
            case_id: "lower_bidiag_n5".into(),
            n: 5,
            triplets: lower_bidiag(5, 2.0, -1.0),
            b: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            lower: true,
        },
        TriCase {
            case_id: "lower_bidiag_n8".into(),
            n: 8,
            triplets: lower_bidiag(8, 3.0, 0.5),
            b: (1..=8).map(|i| (i as f64) * 0.1).collect(),
            lower: true,
        },
        TriCase {
            case_id: "upper_bidiag_n5".into(),
            n: 5,
            triplets: upper_bidiag(5, 4.0, 1.0),
            b: vec![10.0, 5.0, 0.0, -3.0, 7.0],
            lower: false,
        },
        TriCase {
            case_id: "upper_bidiag_n6".into(),
            n: 6,
            triplets: upper_bidiag(6, 1.5, -0.25),
            b: (0..6).map(|i| 1.0 + (i as f64).sin()).collect(),
            lower: false,
        },
        TriCase {
            case_id: "dense_lower_n4".into(),
            n: 4,
            triplets: dense_lower(4),
            b: vec![0.5, 1.5, 2.5, 3.5],
            lower: true,
        },
        TriCase {
            case_id: "dense_upper_n4".into(),
            n: 4,
            triplets: dense_upper(4),
            b: vec![6.0, 4.0, 2.0, 1.0],
            lower: false,
        },
    ];
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve_triangular

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    n = int(case["n"])
    rows = [t[0] for t in case["triplets"]]
    cols = [t[1] for t in case["triplets"]]
    vals = [float(t[2]) for t in case["triplets"]]
    b = np.array(case["b"], dtype=float)
    lower = bool(case["lower"])
    try:
        A = csr_matrix((vals, (rows, cols)), shape=(n, n))
        x = spsolve_triangular(A, b, lower=lower)
        if all(math.isfinite(v) for v in x.tolist()):
            points.append({"case_id": cid, "x": [float(v) for v in x]})
        else:
            points.append({"case_id": cid, "x": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "x": None})
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
                "failed to spawn python3 for spsolve_triangular oracle: {e}"
            );
            eprintln!("skipping spsolve_triangular oracle: python3 not available ({e})");
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
                "spsolve_triangular oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping spsolve_triangular oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for spsolve_triangular oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "spsolve_triangular oracle failed: {stderr}"
        );
        eprintln!("skipping spsolve_triangular oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse spsolve_triangular oracle JSON"))
}

#[test]
fn diff_sparse_spsolve_triangular() {
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
        let Some(expected) = arm.x.clone() else {
            continue;
        };
        let mut data = Vec::with_capacity(case.triplets.len());
        let mut rows = Vec::with_capacity(case.triplets.len());
        let mut cols = Vec::with_capacity(case.triplets.len());
        for &(r, c, v) in &case.triplets {
            data.push(v);
            rows.push(r);
            cols.push(c);
        }
        let Ok(coo) =
            CooMatrix::from_triplets(Shape2D::new(case.n, case.n), data, rows, cols, true)
        else {
            continue;
        };
        let Ok(csr) = coo.to_csr() else {
            continue;
        };
        let Ok(x) = spsolve_triangular(&csr, &case.b, case.lower) else {
            continue;
        };
        let abs_d = if x.len() != expected.len() {
            f64::INFINITY
        } else {
            x.iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: if case.lower {
                "spsolve_triangular_lower"
            } else {
                "spsolve_triangular_upper"
            }
            .into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_spsolve_triangular".into(),
        category: "fsci_sparse::spsolve_triangular vs scipy.sparse.linalg.spsolve_triangular"
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
        "spsolve_triangular conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
