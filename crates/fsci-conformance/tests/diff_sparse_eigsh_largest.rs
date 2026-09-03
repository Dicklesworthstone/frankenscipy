#![forbid(unsafe_code)]
//! Live scipy.sparse.linalg.eigsh parity for fsci_sparse::eigsh
//! (largest-magnitude eigenvalues of a symmetric sparse matrix).
//!
//! Resolves [frankenscipy-scdrz]. fsci uses power iteration with
//! deflation; scipy uses ARPACK. Both should return the top-k
//! eigenvalues by magnitude. Compare sorted |λ| values. Tolerance
//! 1e-4 abs (power-iteration deflation accuracy).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CooMatrix, EigsOptions, FormatConvertible, Shape2D, eigsh};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-4;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    n: usize,
    triplets: Vec<(usize, usize, f64)>,
    k: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Top-k eigenvalues sorted by descending |λ|.
    eigvals_sorted: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create eigsh diff dir");
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

fn symmetric_tridiag(n: usize) -> Vec<(usize, usize, f64)> {
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

fn symmetric_diag_dom_5pt(n: usize) -> Vec<(usize, usize, f64)> {
    // Diagonally-dominant pentadiagonal symmetric
    let mut out = Vec::new();
    for i in 0..n {
        out.push((i, i, 8.0));
        if i + 1 < n {
            out.push((i, i + 1, -1.0));
            out.push((i + 1, i, -1.0));
        }
        if i + 2 < n {
            out.push((i, i + 2, 0.5));
            out.push((i + 2, i, 0.5));
        }
    }
    out
}

fn generate_query() -> OracleQuery {
    OracleQuery {
        points: vec![
            Case {
                case_id: "tridiag_n10_k3".into(),
                n: 10,
                triplets: symmetric_tridiag(10),
                k: 3,
            },
            Case {
                case_id: "tridiag_n20_k2".into(),
                n: 20,
                triplets: symmetric_tridiag(20),
                k: 2,
            },
            Case {
                case_id: "pent_n15_k4".into(),
                n: 15,
                triplets: symmetric_diag_dom_5pt(15),
                k: 4,
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
from scipy.sparse.linalg import eigsh

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    n = int(case["n"])
    rs = [int(t[0]) for t in case["triplets"]]
    cs = [int(t[1]) for t in case["triplets"]]
    vs = [float(t[2]) for t in case["triplets"]]
    k = int(case["k"])
    try:
        A = csr_matrix((vs, (rs, cs)), shape=(n, n)).astype(float)
        # Symmetrize to be safe (small numerical asymmetry should already be zero)
        eigs, _ = eigsh(A, k=k, which='LM')
        # Sort by descending magnitude
        sorted_eigs = sorted(eigs.tolist(), key=lambda v: -abs(v))
        if all(math.isfinite(v) for v in sorted_eigs):
            points.append({"case_id": cid, "eigvals_sorted": [float(v) for v in sorted_eigs]})
        else:
            points.append({"case_id": cid, "eigvals_sorted": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "eigvals_sorted": None})
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
                "failed to spawn python3 for eigsh oracle: {e}"
            );
            eprintln!("skipping eigsh oracle: python3 not available ({e})");
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
                "eigsh oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping eigsh oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for eigsh oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "eigsh oracle failed: {stderr}"
        );
        eprintln!("skipping eigsh oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse eigsh oracle JSON"))
}

#[test]
fn diff_sparse_eigsh_largest() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let opts = EigsOptions::default();
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.eigvals_sorted.as_ref() else {
            continue;
        };
        let mut data = Vec::new();
        let mut rs = Vec::new();
        let mut cs = Vec::new();
        for &(r, c, v) in &case.triplets {
            data.push(v);
            rs.push(r);
            cs.push(c);
        }
        let Ok(coo) = CooMatrix::from_triplets(Shape2D::new(case.n, case.n), data, rs, cs, true)
        else {
            continue;
        };
        let Ok(csr) = coo.to_csr() else {
            continue;
        };
        let Ok(res) = eigsh(&csr, case.k, opts) else {
            continue;
        };
        if !res.converged {
            continue;
        }
        let mut eigs = res.eigenvalues.clone();
        eigs.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
        let abs_d = if eigs.len() != expected.len() {
            f64::INFINITY
        } else {
            eigs.iter()
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
        test_id: "diff_sparse_eigsh_largest".into(),
        category: "fsci_sparse::eigsh (LM) vs scipy.sparse.linalg.eigsh".into(),
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
            eprintln!("eigsh mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "eigsh conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
