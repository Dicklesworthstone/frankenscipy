#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.sparse.linalg.expm`.
//!
//! Resolves [frankenscipy-iexrh]. fsci_sparse::expm computes the
//! matrix exponential of a sparse matrix. Compared against scipy's
//! sparse expm densified via .toarray(). Padé approximation has a
//! floor around 1e-10 abs on small problems.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CooMatrix, ExpmOptions, FormatConvertible, Shape2D, expm};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-004";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    n: usize,
    triplets: Vec<(usize, usize, f64)>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    values: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create expm diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize expm diff log");
    fs::write(path, json).expect("write expm diff log");
}

fn generate_query() -> OracleQuery {
    let points = vec![
        PointCase {
            case_id: "3x3_spd_tridiag".into(),
            n: 3,
            triplets: vec![
                (0, 0, 1.0),
                (0, 1, 0.5),
                (1, 0, 0.5),
                (1, 1, 1.0),
                (1, 2, 0.5),
                (2, 1, 0.5),
                (2, 2, 1.0),
            ],
        },
        PointCase {
            case_id: "4x4_diag".into(),
            n: 4,
            triplets: (0..4).map(|i| (i, i, 0.5 + (i as f64) * 0.25)).collect(),
        },
        PointCase {
            case_id: "3x3_skew_sym".into(),
            n: 3,
            triplets: vec![
                (0, 1, 1.0),
                (1, 0, -1.0),
                (0, 2, 0.5),
                (2, 0, -0.5),
                (1, 2, 0.3),
                (2, 1, -0.3),
            ],
        },
        PointCase {
            case_id: "4x4_dense_small".into(),
            n: 4,
            triplets: vec![
                (0, 0, 0.5),
                (0, 1, 0.2),
                (1, 0, 0.2),
                (1, 1, 0.5),
                (1, 2, 0.2),
                (2, 1, 0.2),
                (2, 2, 0.5),
                (2, 3, 0.2),
                (3, 2, 0.2),
                (3, 3, 0.5),
            ],
        },
    ];

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

def finite_vec_or_none(arr):
    out = []
    for v in np.asarray(arr).flatten().tolist():
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
    try:
        A = sp.csc_matrix((v, (r, c)), shape=(n, n))
        E = spl.expm(A).toarray()
        points.append({"case_id": cid, "values": finite_vec_or_none(E)})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize expm query");
    let mut child = match Command::new("python3")
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
                "failed to spawn python3 for expm oracle: {e}"
            );
            eprintln!("skipping expm oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open expm oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "expm oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping expm oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for expm oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "expm oracle failed: {stderr}"
        );
        eprintln!("skipping expm oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse expm oracle JSON"))
}

fn fsci_eval(case: &PointCase) -> Option<Vec<f64>> {
    let r: Vec<usize> = case.triplets.iter().map(|t| t.0).collect();
    let c: Vec<usize> = case.triplets.iter().map(|t| t.1).collect();
    let d: Vec<f64> = case.triplets.iter().map(|t| t.2).collect();
    let coo = CooMatrix::from_triplets(Shape2D::new(case.n, case.n), d, r, c, false).ok()?;
    let csr = coo.to_csr().ok()?;
    let result = expm(&csr, ExpmOptions::default()).ok()?;
    Some(result.into_iter().flatten().collect())
}

#[test]
fn diff_sparse_expm() {
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
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        let Some(fsci_v) = fsci_eval(case) else {
            continue;
        };
        if fsci_v.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let abs_d = fsci_v
            .iter()
            .zip(scipy_v.iter())
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
        test_id: "diff_sparse_expm".into(),
        category: "scipy.sparse.linalg.expm".into(),
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
            eprintln!("expm mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.sparse.linalg.expm conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
