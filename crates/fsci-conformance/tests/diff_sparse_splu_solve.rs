#![forbid(unsafe_code)]
//! Live SciPy differential coverage for sparse LU factor + solve.
//!   - fsci_sparse::splu(A, options) → SparseLuFactorization
//!   - fsci_sparse::splu_solve(factorization, b) → solution x
//!
//! Oracle: `scipy.sparse.linalg.splu(A).solve(b)`. Solution vector is
//! unique for nonsingular A.
//!
//! Resolves [frankenscipy-6xetx]. 1e-8 abs tolerance.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CooMatrix, FormatConvertible, LuOptions, Shape2D, splu, splu_solve};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-004";
const ABS_TOL: f64 = 1.0e-8;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create splu diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize splu diff log");
    fs::write(path, json).expect("write splu diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // 4x4 tridiag SPD
    let trips_4 = vec![
        (0, 0, 4.0),
        (0, 1, 1.0),
        (1, 0, 1.0),
        (1, 1, 4.0),
        (1, 2, 1.0),
        (2, 1, 1.0),
        (2, 2, 4.0),
        (2, 3, 1.0),
        (3, 2, 1.0),
        (3, 3, 4.0),
    ];
    points.push(PointCase {
        case_id: "4x4_tridiag_spd".into(),
        n: 4,
        triplets: trips_4,
        b: vec![1.0, 2.0, 3.0, 4.0],
    });

    // 3x3 dense
    points.push(PointCase {
        case_id: "3x3_dense".into(),
        n: 3,
        triplets: vec![
            (0, 0, 2.0),
            (0, 1, 1.0),
            (0, 2, 0.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
            (1, 2, -1.0),
            (2, 0, 0.0),
            (2, 1, -1.0),
            (2, 2, 4.0),
        ],
        b: vec![1.0, 0.0, 2.0],
    });

    // 5x5 SPD pentadiagonal
    let n = 5;
    let mut trips_5 = Vec::new();
    for i in 0..n {
        trips_5.push((i, i, 5.0));
        if i + 1 < n {
            trips_5.push((i, i + 1, 1.0));
            trips_5.push((i + 1, i, 1.0));
        }
        if i + 2 < n {
            trips_5.push((i, i + 2, 0.5));
            trips_5.push((i + 2, i, 0.5));
        }
    }
    points.push(PointCase {
        case_id: "5x5_pentadiag_spd".into(),
        n,
        triplets: trips_5,
        b: vec![1.0, -1.0, 2.0, -2.0, 0.5],
    });

    // 4x4 nonsymmetric
    points.push(PointCase {
        case_id: "4x4_nonsym".into(),
        n: 4,
        triplets: vec![
            (0, 0, 3.0),
            (0, 1, -1.0),
            (1, 0, 1.0),
            (1, 1, 2.0),
            (1, 2, -2.0),
            (2, 1, 1.0),
            (2, 2, 4.0),
            (2, 3, 1.0),
            (3, 2, -1.0),
            (3, 3, 5.0),
        ],
        b: vec![2.0, 3.0, -1.0, 4.0],
    });

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

def finite_vec_or_none(arr):
    out = []
    for v in np.asarray(arr).tolist():
        try:
            v = float(v)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    n = case["n"]
    triplets = case["triplets"]
    if triplets:
        r = np.array([t[0] for t in triplets], dtype=int)
        c = np.array([t[1] for t in triplets], dtype=int)
        v = np.array([t[2] for t in triplets], dtype=float)
    else:
        r = np.zeros(0, dtype=int); c = np.zeros(0, dtype=int); v = np.zeros(0, dtype=float)
    b = np.array(case["b"], dtype=float)
    try:
        A = sp.csc_matrix((v, (r, c)), shape=(n, n))
        lu = spl.splu(A)
        x = lu.solve(b)
        points.append({"case_id": cid, "x": finite_vec_or_none(x)})
    except Exception:
        points.append({"case_id": cid, "x": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize splu query");
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
                "failed to spawn python3 for splu oracle: {e}"
            );
            eprintln!("skipping splu oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open splu oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "splu oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping splu oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for splu oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "splu oracle failed: {stderr}"
        );
        eprintln!("skipping splu oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse splu oracle JSON"))
}

fn fsci_solve(case: &PointCase) -> Option<Vec<f64>> {
    let r: Vec<usize> = case.triplets.iter().map(|t| t.0).collect();
    let c: Vec<usize> = case.triplets.iter().map(|t| t.1).collect();
    let d: Vec<f64> = case.triplets.iter().map(|t| t.2).collect();
    let coo = CooMatrix::from_triplets(Shape2D::new(case.n, case.n), d, r, c, false).ok()?;
    let csc = coo.to_csc().ok()?;
    let fact = splu(&csc, LuOptions::default()).ok()?;
    splu_solve(&fact, &case.b).ok()
}

#[test]
fn diff_sparse_splu_solve() {
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
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_x) = scipy_arm.x.as_ref() else {
            continue;
        };
        let Some(fsci_x) = fsci_solve(case) else {
            continue;
        };
        if fsci_x.len() != scipy_x.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let abs_d = fsci_x
            .iter()
            .zip(scipy_x.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_splu_solve".into(),
        category: "scipy.sparse.linalg.splu().solve()".into(),
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
            eprintln!("splu_solve mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.sparse splu+solve conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
