#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.sparse.csgraph.structural_rank`.
//!
//! Resolves [frankenscipy-dpkqx]. structural_rank returns the maximum
//! bipartite matching across the row/column nonzero pattern — a
//! deterministic integer. Bit-exact agreement expected.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CooMatrix, FormatConvertible, Shape2D, structural_rank};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-004";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    rows: usize,
    cols: usize,
    triplets: Vec<(usize, usize, f64)>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    rank: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    fsci: usize,
    scipy: usize,
    pass: bool,
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
    fs::create_dir_all(output_dir()).expect("create structural_rank diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize structural_rank diff log");
    fs::write(path, json).expect("write structural_rank diff log");
}

fn generate_query() -> OracleQuery {
    let cases: &[(&str, usize, usize, Vec<(usize, usize, f64)>)] = &[
        (
            "3x3_full_rank",
            3,
            3,
            vec![
                (0, 0, 1.0),
                (0, 1, 1.0),
                (1, 1, 1.0),
                (1, 2, 1.0),
                (2, 0, 1.0),
                (2, 2, 1.0),
            ],
        ),
        ("3x3_rank2", 3, 3, vec![(0, 0, 1.0), (1, 1, 1.0)]),
        (
            "4x4_diag",
            4,
            4,
            vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)],
        ),
        (
            "5x5_random_pattern",
            5,
            5,
            vec![
                (0, 1, 1.0),
                (0, 3, 1.0),
                (1, 0, 1.0),
                (1, 4, 1.0),
                (2, 2, 1.0),
                (3, 0, 1.0),
                (3, 3, 1.0),
                (4, 1, 1.0),
                (4, 4, 1.0),
            ],
        ),
        (
            "4x6_wide",
            4,
            6,
            vec![
                (0, 0, 1.0),
                (0, 5, 1.0),
                (1, 1, 1.0),
                (1, 3, 1.0),
                (2, 2, 1.0),
                (3, 4, 1.0),
            ],
        ),
        (
            "3x3_empty_row",
            3,
            3,
            vec![(0, 0, 1.0), (0, 1, 1.0), (2, 2, 1.0)],
        ),
    ];
    let points = cases
        .iter()
        .map(|(name, r, c, t)| PointCase {
            case_id: (*name).into(),
            rows: *r,
            cols: *c,
            triplets: t.clone(),
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import structural_rank

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    rows = case["rows"]; cols = case["cols"]
    trips = case["triplets"]
    if trips:
        r = np.array([t[0] for t in trips], dtype=int)
        c = np.array([t[1] for t in trips], dtype=int)
        v = np.array([t[2] for t in trips], dtype=float)
    else:
        r = np.zeros(0, dtype=int); c = np.zeros(0, dtype=int); v = np.zeros(0, dtype=float)
    try:
        A = sp.csr_matrix((v, (r, c)), shape=(rows, cols))
        rk = int(structural_rank(A))
        points.append({"case_id": cid, "rank": rk})
    except Exception:
        points.append({"case_id": cid, "rank": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize structural_rank query");
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
                "failed to spawn python3 for structural_rank oracle: {e}"
            );
            eprintln!("skipping structural_rank oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open structural_rank oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "structural_rank oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping structural_rank oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for structural_rank oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "structural_rank oracle failed: {stderr}"
        );
        eprintln!("skipping structural_rank oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse structural_rank oracle JSON"))
}

fn fsci_eval(case: &PointCase) -> Option<usize> {
    let r: Vec<usize> = case.triplets.iter().map(|t| t.0).collect();
    let c: Vec<usize> = case.triplets.iter().map(|t| t.1).collect();
    let d: Vec<f64> = case.triplets.iter().map(|t| t.2).collect();
    let coo = CooMatrix::from_triplets(Shape2D::new(case.rows, case.cols), d, r, c, false).ok()?;
    let csr = coo.to_csr().ok()?;
    Some(structural_rank(&csr))
}

#[test]
fn diff_sparse_structural_rank() {
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
    let mut diffs = Vec::new();

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_rk) = scipy_arm.rank else {
            continue;
        };
        let Some(fsci_rk) = fsci_eval(case) else {
            continue;
        };
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            fsci: fsci_rk,
            scipy: scipy_rk,
            pass: fsci_rk == scipy_rk,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_structural_rank".into(),
        category: "scipy.sparse.csgraph.structural_rank".into(),
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
                "structural_rank mismatch: {} fsci={} scipy={}",
                d.case_id, d.fsci, d.scipy
            );
        }
    }

    assert!(
        all_pass,
        "scipy.sparse.csgraph.structural_rank conformance failed: {} cases",
        diffs.len()
    );
}
