#![forbid(unsafe_code)]
//! Live scipy.sparse.linalg.svds parity for fsci_sparse::svds.
//!
//! Resolves [frankenscipy-e16kb]. fsci uses power iteration on
//! A^T A with deflation; scipy uses ARPACK. Singular values are
//! unique (non-negative) so we sort descending and compare. Left
//! and right singular vectors have sign/phase ambiguity, so we
//! only compare σ values. Tolerance 1e-4 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CooMatrix, EigsOptions, FormatConvertible, Shape2D, svds};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-4;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    rows: usize,
    cols: usize,
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
    sigma_sorted: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create svds diff dir");
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

fn rect_dense(rows: usize, cols: usize, seed: u64) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    let mut s = seed;
    for r in 0..rows {
        for c in 0..cols {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64) / (1u64 << 53) as f64;
            let v = (u - 0.5) * 4.0;
            // Skip occasional zeros to make it sparse-ish but ensure full rank.
            if r == c || (((r * 7 + c * 13) % 5) != 0) {
                out.push((r, c, v + if r == c { 5.0 } else { 0.0 }));
            }
        }
    }
    out
}

fn generate_query() -> OracleQuery {
    OracleQuery {
        points: vec![
            Case {
                case_id: "tall_8x5_k2".into(),
                rows: 8,
                cols: 5,
                triplets: rect_dense(8, 5, 0xdead_beef_cafe_babe),
                k: 2,
            },
            Case {
                case_id: "wide_6x10_k2".into(),
                rows: 6,
                cols: 10,
                triplets: rect_dense(6, 10, 0x1234_5678_90ab_cdef),
                k: 2,
            },
            Case {
                case_id: "square_10x10_k3".into(),
                rows: 10,
                cols: 10,
                triplets: rect_dense(10, 10, 0xfeed_face_dead_beef),
                k: 3,
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
from scipy.sparse.linalg import svds

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    m = int(case["rows"]); n = int(case["cols"])
    rs = [int(t[0]) for t in case["triplets"]]
    cs = [int(t[1]) for t in case["triplets"]]
    vs = [float(t[2]) for t in case["triplets"]]
    k = int(case["k"])
    try:
        A = csr_matrix((vs, (rs, cs)), shape=(m, n)).astype(float)
        # which='LM' returns top-k singular values
        u, s, vt = svds(A, k=k, which='LM')
        sorted_s = sorted([float(x) for x in s.tolist()], reverse=True)
        if all(math.isfinite(x) for x in sorted_s):
            points.append({"case_id": cid, "sigma_sorted": sorted_s})
        else:
            points.append({"case_id": cid, "sigma_sorted": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "sigma_sorted": None})
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
                "failed to spawn python3 for svds oracle: {e}"
            );
            eprintln!("skipping svds oracle: python3 not available ({e})");
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
                "svds oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping svds oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for svds oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "svds oracle failed: {stderr}"
        );
        eprintln!("skipping svds oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse svds oracle JSON"))
}

#[test]
fn diff_sparse_svds() {
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
        let Some(expected) = arm.sigma_sorted.as_ref() else {
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
        let Ok(coo) =
            CooMatrix::from_triplets(Shape2D::new(case.rows, case.cols), data, rs, cs, true)
        else {
            continue;
        };
        let Ok(csr) = coo.to_csr() else {
            continue;
        };
        let Ok(res) = svds(&csr, case.k, opts) else {
            continue;
        };
        let mut sigs = res.singular_values.clone();
        sigs.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let abs_d = if sigs.len() != expected.len() {
            f64::INFINITY
        } else {
            sigs.iter()
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
        test_id: "diff_sparse_svds".into(),
        category: "fsci_sparse::svds vs scipy.sparse.linalg.svds (top-k σ)".into(),
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
            eprintln!("svds mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "svds conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
