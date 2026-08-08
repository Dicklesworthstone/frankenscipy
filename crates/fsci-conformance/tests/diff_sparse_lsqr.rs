#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.sparse.linalg.lsqr`.
//!
//! Resolves [frankenscipy-itrec]. fsci_sparse::lsqr solves Ax ≈ b in
//! the least-squares sense. Probe on over-determined rectangular
//! systems where the least-squares solution is unique. 1e-5 abs
//! tolerance (iterative bidiag floor).
//!
//! This file used to state that "lsmr delegates to lsqr in fsci". That
//! was true when written and is now false: `lsmr` is a genuine Fong &
//! Saunders second-QR recurrence (frankenscipy-6pdfn) and carries its
//! own live-SciPy column in `diff_sparse_iterative_solvers`. Only
//! `lsqr` is under test here.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CooMatrix, FormatConvertible, IterativeSolveOptions, Shape2D, lsqr};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-004";
const ABS_TOL: f64 = 1.0e-5;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    rows: usize,
    cols: usize,
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
    fs::create_dir_all(output_dir()).expect("create lsqr diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize lsqr diff log");
    fs::write(path, json).expect("write lsqr diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // 5x3 over-determined, full column rank
    points.push(PointCase {
        case_id: "5x3_overdetermined".into(),
        rows: 5,
        cols: 3,
        triplets: vec![
            (0, 0, 1.0),
            (0, 1, 0.5),
            (1, 0, 0.5),
            (1, 1, 1.0),
            (1, 2, 0.5),
            (2, 1, 0.5),
            (2, 2, 1.0),
            (3, 0, 1.0),
            (3, 2, -0.5),
            (4, 1, 1.0),
            (4, 2, 1.0),
        ],
        b: vec![1.0, 2.0, 3.0, 1.5, 2.5],
    });

    // 6x4 over-determined with diagonal dominance
    points.push(PointCase {
        case_id: "6x4_diag_dom".into(),
        rows: 6,
        cols: 4,
        triplets: vec![
            (0, 0, 3.0),
            (1, 1, 3.0),
            (2, 2, 3.0),
            (3, 3, 3.0),
            (4, 0, 1.0),
            (4, 1, 1.0),
            (5, 2, 1.0),
            (5, 3, 1.0),
        ],
        b: vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
    });

    // 4x4 square nonsingular (lsqr converges to A^{-1} b)
    points.push(PointCase {
        case_id: "4x4_square_nonsingular".into(),
        rows: 4,
        cols: 4,
        triplets: vec![
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
        ],
        b: vec![1.0, 2.0, 3.0, 4.0],
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
    rows = case["rows"]; cols = case["cols"]
    trips = case["triplets"]
    if trips:
        r = np.array([t[0] for t in trips], dtype=int)
        c = np.array([t[1] for t in trips], dtype=int)
        v = np.array([t[2] for t in trips], dtype=float)
    else:
        r = np.zeros(0, dtype=int); c = np.zeros(0, dtype=int); v = np.zeros(0, dtype=float)
    b = np.array(case["b"], dtype=float)
    try:
        A = sp.csr_matrix((v, (r, c)), shape=(rows, cols))
        res = spl.lsqr(A, b, atol=1e-10, btol=1e-10, iter_lim=500)
        x = res[0]
        points.append({"case_id": cid, "x": finite_vec_or_none(x)})
    except Exception:
        points.append({"case_id": cid, "x": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize lsqr query");
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
                "failed to spawn python3 for lsqr oracle: {e}"
            );
            eprintln!("skipping lsqr oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open lsqr oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "lsqr oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping lsqr oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for lsqr oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "lsqr oracle failed: {stderr}"
        );
        eprintln!("skipping lsqr oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse lsqr oracle JSON"))
}

fn fsci_eval(case: &PointCase) -> Option<Vec<f64>> {
    let r: Vec<usize> = case.triplets.iter().map(|t| t.0).collect();
    let c: Vec<usize> = case.triplets.iter().map(|t| t.1).collect();
    let d: Vec<f64> = case.triplets.iter().map(|t| t.2).collect();
    let coo = CooMatrix::from_triplets(Shape2D::new(case.rows, case.cols), d, r, c, false).ok()?;
    let csr = coo.to_csr().ok()?;
    let opts = IterativeSolveOptions {
        tol: 1.0e-10,
        max_iter: Some(500),
        ..Default::default()
    };
    let res = lsqr(&csr, &case.b, opts).ok()?;
    Some(res.solution)
}

#[test]
fn diff_sparse_lsqr() {
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

    // A case that either arm fails to solve used to `continue` silently, so a
    // harness that compared nothing still reported "pass". Skips are now
    // recorded and asserted on below.
    let mut skipped: Vec<String> = Vec::new();

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_x) = scipy_arm.x.as_ref() else {
            skipped.push(format!("{} (scipy arm returned no solution)", case.case_id));
            continue;
        };
        let Some(fsci_x) = fsci_eval(case) else {
            skipped.push(format!("{} (fsci arm did not converge)", case.case_id));
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

    // The oracle arm must have genuinely compared at least one case. Without
    // this, an environment where every scipy call raises (no scipy installed,
    // an unsupported kwarg) contributes zero cases and `all_pass` is vacuously
    // true — a green report over a column that tested nothing.
    assert!(
        !diffs.is_empty(),
        "lsqr compared zero cases; this harness cannot pass by testing nothing. \
         Skipped {} of {} case(s): {skipped:?}",
        skipped.len(),
        query.points.len()
    );
    if !skipped.is_empty() {
        eprintln!("lsqr skipped {} case(s): {skipped:?}", skipped.len());
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_lsqr".into(),
        category: "scipy.sparse.linalg.lsqr".into(),
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
            eprintln!("lsqr mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.sparse.linalg.lsqr conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
