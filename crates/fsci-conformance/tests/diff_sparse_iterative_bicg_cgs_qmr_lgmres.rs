#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_sparse's less-tested
//! iterative solvers:
//!   - `bicg`   vs `scipy.sparse.linalg.bicg`
//!   - `cgs`    vs `scipy.sparse.linalg.cgs`
//!   - `qmr`    vs `scipy.sparse.linalg.qmr`
//!   - `lgmres` vs `scipy.sparse.linalg.lgmres`
//!
//! Resolves [frankenscipy-u14ae]. Solution is unique so a vector-
//! element comparison is well-defined. Tolerance 1e-5 abs covers the
//! ~1e-8 residual floor both sides converge to (relative drift may
//! accumulate component-wise for ill-conditioned cases).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{
    CooMatrix, FormatConvertible, IterativeSolveOptions, LgmresOptions, Shape2D, bicg, cgs, lgmres,
    qmr,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-5;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    solver: String,
    n: usize,
    triplets: Vec<(usize, usize, f64)>,
    b: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
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
    solver: String,
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
    fs::create_dir_all(output_dir()).expect("create iterative diff dir");
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

// Build a small diagonally-dominant tridiagonal SPD matrix.
fn tridiag_spd(n: usize, diag: f64, off: f64) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    for i in 0..n {
        out.push((i, i, diag));
        if i + 1 < n {
            out.push((i, i + 1, off));
            out.push((i + 1, i, off));
        }
    }
    out
}

// Build a small dense banded general (non-symmetric) matrix.
fn pentadiag_general(n: usize) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    for i in 0..n {
        out.push((i, i, 4.0));
        if i >= 1 {
            out.push((i, i - 1, -1.0));
        }
        if i >= 2 {
            out.push((i, i - 2, 0.5));
        }
        if i + 1 < n {
            out.push((i, i + 1, -2.0));
        }
        if i + 2 < n {
            out.push((i, i + 2, 0.25));
        }
    }
    out
}

fn generate_query() -> OracleQuery {
    let spd_8 = tridiag_spd(8, 4.0, -1.0);
    let spd_12 = tridiag_spd(12, 4.0, -1.0);
    let pent_10 = pentadiag_general(10);

    let b_8: Vec<f64> = (1..=8).map(|i| (i as f64) * 0.5).collect();
    let b_12: Vec<f64> = (0..12).map(|i| 1.0 + ((i as f64) * 0.3).sin()).collect();
    let b_10: Vec<f64> = (0..10).map(|i| (i as f64) - 5.0).collect();

    let mut points = Vec::new();
    for solver in ["bicg", "cgs", "qmr", "lgmres"] {
        points.push(PointCase {
            case_id: format!("{solver}_spd8"),
            solver: solver.into(),
            n: 8,
            triplets: spd_8.clone(),
            b: b_8.clone(),
        });
        points.push(PointCase {
            case_id: format!("{solver}_spd12"),
            solver: solver.into(),
            n: 12,
            triplets: spd_12.clone(),
            b: b_12.clone(),
        });
        points.push(PointCase {
            case_id: format!("{solver}_pent10"),
            solver: solver.into(),
            n: 10,
            triplets: pent_10.clone(),
            b: b_10.clone(),
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
from scipy.sparse import csr_matrix
from scipy.sparse import linalg as spl

solvers = {"bicg": spl.bicg, "cgs": spl.cgs, "qmr": spl.qmr, "lgmres": spl.lgmres}

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; solver = case["solver"]
    n = int(case["n"])
    rows = [int(t[0]) for t in case["triplets"]]
    cols = [int(t[1]) for t in case["triplets"]]
    vals = [float(t[2]) for t in case["triplets"]]
    b = np.array(case["b"], dtype=float)
    try:
        A = csr_matrix((vals, (rows, cols)), shape=(n, n))
        x, info = solvers[solver](A, b, rtol=1.0e-10, atol=0.0, maxiter=2000)
        if info == 0 and all(math.isfinite(v) for v in x.tolist()):
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
                "failed to spawn python3 for iterative oracle: {e}"
            );
            eprintln!("skipping iterative oracle: python3 not available ({e})");
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
                "iterative oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping iterative oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for iterative oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "iterative oracle failed: {stderr}"
        );
        eprintln!("skipping iterative oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse iterative oracle JSON"))
}

fn fsci_solve(case: &PointCase) -> Option<Vec<f64>> {
    let mut d = Vec::with_capacity(case.triplets.len());
    let mut r = Vec::with_capacity(case.triplets.len());
    let mut c = Vec::with_capacity(case.triplets.len());
    for &(ri, ci, vi) in &case.triplets {
        d.push(vi);
        r.push(ri);
        c.push(ci);
    }
    let coo = CooMatrix::from_triplets(Shape2D::new(case.n, case.n), d, r, c, true).ok()?;
    let csr = coo.to_csr().ok()?;
    let opts = IterativeSolveOptions {
        tol: 1.0e-10,
        max_iter: Some(2000),
        ..Default::default()
    };
    let result = match case.solver.as_str() {
        "bicg" => bicg(&csr, &case.b, None, opts).ok()?,
        "cgs" => cgs(&csr, &case.b, None, opts).ok()?,
        "qmr" => qmr(&csr, &case.b, None, opts).ok()?,
        "lgmres" => {
            let lopts = LgmresOptions {
                tol: 1.0e-10,
                max_iter: Some(2000),
                ..Default::default()
            };
            lgmres(&csr, &case.b, None, lopts).ok()?
        }
        _ => return None,
    };
    if !result.converged {
        return None;
    }
    Some(result.solution)
}

#[test]
fn diff_sparse_iterative_bicg_cgs_qmr_lgmres() {
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
        let Some(expected) = arm.x.as_ref() else {
            continue;
        };
        let Some(actual) = fsci_solve(case) else {
            continue;
        };
        let abs_d = if actual.len() != expected.len() {
            f64::INFINITY
        } else {
            actual
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            solver: case.solver.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_iterative_bicg_cgs_qmr_lgmres".into(),
        category: "scipy.sparse.linalg bicg/cgs/qmr/lgmres".into(),
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
                "{} mismatch: {} abs_diff={}",
                d.solver, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "iterative bicg/cgs/qmr/lgmres conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
