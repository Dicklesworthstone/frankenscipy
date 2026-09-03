#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_spatial nearest-
//! neighbor query helpers.
//!
//! Resolves [frankenscipy-vn2v0]. Covers:
//!   • nearest_neighbors(data)
//!     ≡ scipy.spatial.KDTree(data).query(data, k=2)[..., 1]
//!       (k=2 because k=1 returns self at distance 0; idx 1 is
//!       the actual nearest neighbor)
//!   • k_nearest_neighbors(data, k)
//!     ≡ scipy.spatial.KDTree(data).query(data, k=k+1)[..., 1:]
//!
//! 4 fixtures with strictly distinct pairwise distances → no
//! tie-breaking divergence. Indices must match exactly,
//! distances at 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_spatial::{k_nearest_neighbors, nearest_neighbors};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-011";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    data: Vec<Vec<f64>>,
    k: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    nn_idx: Option<Vec<i64>>,
    nn_dist: Option<Vec<f64>>,
    knn_idx: Option<Vec<Vec<i64>>>,
    knn_dist: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    sub_check: String,
    detail: String,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create neighbors diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize neighbors diff log");
    fs::write(path, json).expect("write neighbors diff log");
}

fn generate_query() -> OracleQuery {
    OracleQuery {
        points: vec![
            // 4 points in 2-D, all distances distinct
            PointCase {
                case_id: "n4_2d".into(),
                data: vec![
                    vec![0.0, 0.0],
                    vec![1.0, 0.0],
                    vec![0.0, 3.0],
                    vec![5.0, 7.0],
                ],
                k: 2,
            },
            // 6 points in 3-D
            PointCase {
                case_id: "n6_3d".into(),
                data: vec![
                    vec![0.0, 0.0, 0.0],
                    vec![1.0, 0.5, 0.25],
                    vec![3.0, 1.5, 0.75],
                    vec![0.5, 4.0, 1.0],
                    vec![5.0, 5.5, 6.5],
                    vec![10.0, 0.0, 8.0],
                ],
                k: 3,
            },
            // 8 points in 1-D with strictly increasing gaps so every
            // pairwise distance is distinct (a uniform-spaced chain
            // produces equidistant neighbor ties at interior points
            // that fsci and scipy break in different orders).
            PointCase {
                case_id: "n8_1d_increasing_gaps".into(),
                data: (0..8)
                    .map(|i| vec![(i as f64) * (i as f64 + 1.0) * 0.5])
                    .collect(),
                k: 2,
            },
            // 10 points scattered 2-D
            PointCase {
                case_id: "n10_2d".into(),
                data: vec![
                    vec![0.1, 0.2],
                    vec![0.5, 1.0],
                    vec![1.7, 2.2],
                    vec![3.4, 0.8],
                    vec![4.0, 3.5],
                    vec![5.5, 1.0],
                    vec![6.7, 5.5],
                    vec![8.0, 2.0],
                    vec![9.0, 7.5],
                    vec![10.5, 3.0],
                ],
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
from scipy.spatial import KDTree

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    out = {
        "case_id": cid,
        "nn_idx": None,
        "nn_dist": None,
        "knn_idx": None,
        "knn_dist": None,
    }
    try:
        X = np.asarray(case["data"], dtype=np.float64)
        n = X.shape[0]
        tree = KDTree(X)

        # nearest neighbor: query k=2, take idx 1 (idx 0 is self).
        d2, i2 = tree.query(X, k=2)
        out["nn_idx"] = [int(v) for v in i2[:, 1].tolist()]
        out["nn_dist"] = [fnone(v) for v in d2[:, 1].tolist()]

        # k-nearest neighbors: query k=k+1, drop idx 0.
        kk = case["k"] + 1
        if kk > n:
            kk = n
        dk, ik = tree.query(X, k=kk)
        # When k=2 and shape is (n,2) numpy returns 2-D arrays; for
        # k=1 it would return 1-D. Our fixtures all have k+1 >= 2 so
        # we always have 2-D.
        out["knn_idx"] = [
            [int(v) for v in row[1:].tolist()] for row in ik
        ]
        out["knn_dist"] = [
            [fnone(v) for v in row[1:].tolist()] for row in dk
        ]
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}, allow_nan=False))
"#;
    let query_json = serde_json::to_string(query).expect("serialize neighbors query");
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
                "failed to spawn python3 for neighbors oracle: {e}"
            );
            eprintln!("skipping neighbors oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open neighbors oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "neighbors oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping neighbors oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for neighbors oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "neighbors oracle failed: {stderr}"
        );
        eprintln!("skipping neighbors oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse neighbors oracle JSON"))
}

#[test]
fn diff_spatial_neighbors() {
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
    let mut cases = Vec::new();

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");

        // nearest_neighbors
        if let (Some(scipy_idx), Some(scipy_dist)) =
            (scipy_arm.nn_idx.as_ref(), scipy_arm.nn_dist.as_ref())
        {
            let (rust_idx_opt, rust_dist) = nearest_neighbors(&case.data);
            let rust_idx: Vec<i64> = rust_idx_opt
                .iter()
                .map(|o| o.map_or(-1_i64, |v| v as i64))
                .collect();
            let idx_match = rust_idx == *scipy_idx;
            let dist_pass = rust_dist.len() == scipy_dist.len()
                && rust_dist
                    .iter()
                    .zip(scipy_dist.iter())
                    .all(|(r, s)| (r - s).abs() <= ABS_TOL);
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "nearest_neighbors".into(),
                detail: format!(
                    "idx_match={idx_match}, rust_idx={rust_idx:?}, scipy_idx={scipy_idx:?}"
                ),
                pass: idx_match && dist_pass,
            });
        }

        // k_nearest_neighbors
        if let (Some(scipy_idx), Some(scipy_dist)) =
            (scipy_arm.knn_idx.as_ref(), scipy_arm.knn_dist.as_ref())
        {
            let (rust_idx, rust_dist) = k_nearest_neighbors(&case.data, case.k);
            let idx_match = rust_idx.len() == scipy_idx.len()
                && rust_idx
                    .iter()
                    .zip(scipy_idx.iter())
                    .all(|(r, s)| r.iter().map(|&v| v as i64).collect::<Vec<_>>() == *s);
            let dist_pass = rust_dist.len() == scipy_dist.len()
                && rust_dist.iter().zip(scipy_dist.iter()).all(|(rr, sr)| {
                    rr.len() == sr.len()
                        && rr
                            .iter()
                            .zip(sr.iter())
                            .all(|(r, s)| (r - s).abs() <= ABS_TOL)
                });
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "k_nearest_neighbors".into(),
                detail: format!("idx_match={idx_match}, dist_pass={dist_pass}, k={}", case.k),
                pass: idx_match && dist_pass,
            });
        }
    }

    let pass_count = cases.iter().filter(|c| c.pass).count();
    let all_pass = pass_count == cases.len();

    let log = DiffLog {
        test_id: "diff_spatial_neighbors".into(),
        category: "fsci_spatial::{nearest_neighbors,k_nearest_neighbors}".into(),
        case_count: cases.len(),
        pass_count,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: cases.clone(),
    };

    emit_log(&log);

    for c in &cases {
        if !c.pass {
            eprintln!(
                "neighbors mismatch: {} {} — {}",
                c.case_id, c.sub_check, c.detail
            );
        }
    }

    assert!(
        all_pass,
        "neighbors conformance failed: {} of {} cases pass",
        pass_count,
        cases.len()
    );
}
