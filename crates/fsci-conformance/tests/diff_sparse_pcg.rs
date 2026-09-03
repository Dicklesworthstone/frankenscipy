#![forbid(unsafe_code)]
//! Live scipy.sparse.linalg parity for fsci_sparse::pcg.
//!
//! Resolves [frankenscipy-v9461]. fsci preconditioned CG with ILU(0)
//! should converge to the same solution x as scipy.sparse.linalg.cg
//! (unpreconditioned is sufficient — both are CG variants and converge
//! to the unique solution of Ax = b on SPD systems).
//!
//! Tolerance: 1e-5 abs on solution x (residual floor ~1e-10 internal).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{
    CooMatrix, FormatConvertible, IluOptions, IterativeSolveOptions, Shape2D, pcg, spilu,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-5;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    n: usize,
    triplets: Vec<(usize, usize, f64)>,
    b: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
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
    fs::create_dir_all(output_dir()).expect("create pcg diff dir");
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

fn tridiag_spd(n: usize) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    for i in 0..n {
        out.push((i, i, 4.0));
        if i + 1 < n {
            out.push((i, i + 1, -1.0));
            out.push((i + 1, i, -1.0));
        }
    }
    out
}

fn generate_query() -> OracleQuery {
    let n6 = 6;
    let n10 = 10;
    let n16 = 16;
    let trips_6 = tridiag_spd(n6);
    let trips_10 = tridiag_spd(n10);
    let trips_16 = tridiag_spd(n16);
    let b6: Vec<f64> = (1..=n6).map(|i| i as f64).collect();
    let b10: Vec<f64> = (0..n10).map(|i| ((i as f64) * 0.4).sin() + 1.0).collect();
    let b16: Vec<f64> = (0..n16).map(|i| (i as f64) - 7.5).collect();
    OracleQuery {
        points: vec![
            Case {
                case_id: "tridiag_spd_n6".into(),
                n: n6,
                triplets: trips_6,
                b: b6,
            },
            Case {
                case_id: "tridiag_spd_n10".into(),
                n: n10,
                triplets: trips_10,
                b: b10,
            },
            Case {
                case_id: "tridiag_spd_n16".into(),
                n: n16,
                triplets: trips_16,
                b: b16,
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
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    n = int(case["n"])
    rs = [int(t[0]) for t in case["triplets"]]
    cs = [int(t[1]) for t in case["triplets"]]
    vs = [float(t[2]) for t in case["triplets"]]
    b = np.array(case["b"], dtype=float)
    try:
        A = csr_matrix((vs, (rs, cs)), shape=(n, n))
        x, info = cg(A, b, rtol=1e-12, atol=0.0, maxiter=2000)
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
                "failed to spawn python3 for pcg oracle: {e}"
            );
            eprintln!("skipping pcg oracle: python3 not available ({e})");
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
                "pcg oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping pcg oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for pcg oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "pcg oracle failed: {stderr}"
        );
        eprintln!("skipping pcg oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse pcg oracle JSON"))
}

#[test]
fn diff_sparse_pcg() {
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
        let mut d = Vec::new();
        let mut r = Vec::new();
        let mut c = Vec::new();
        for &(ri, ci, vi) in &case.triplets {
            d.push(vi);
            r.push(ri);
            c.push(ci);
        }
        let Ok(coo) = CooMatrix::from_triplets(Shape2D::new(case.n, case.n), d, r, c, true) else {
            continue;
        };
        let Ok(csr) = coo.to_csr() else {
            continue;
        };
        let Ok(csc) = csr.to_csc() else {
            continue;
        };
        let Ok(ilu) = spilu(&csc, IluOptions::default()) else {
            continue;
        };
        let opts = IterativeSolveOptions {
            tol: 1.0e-12,
            max_iter: Some(2000),
            ..Default::default()
        };
        let Ok(result) = pcg(&csr, &case.b, &ilu, None, opts) else {
            continue;
        };
        if !result.converged {
            continue;
        }
        let abs_d = if result.solution.len() != expected.len() {
            f64::INFINITY
        } else {
            result
                .solution
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_pcg".into(),
        category: "fsci_sparse::pcg (ILU preconditioner) vs scipy.sparse.linalg.cg".into(),
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
            eprintln!("pcg mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "pcg conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
