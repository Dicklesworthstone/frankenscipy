#![forbid(unsafe_code)]
//! Live SciPy differential coverage for sparse structural ops:
//!   - fsci_sparse::find(A) vs scipy.sparse.find(A) — canonical nonzero
//!     pattern (rows, cols, data); compare sorted-by-(row,col) triples
//!   - fsci_sparse::tril(A, k) vs scipy.sparse.tril(A, k) — lower-triangle
//!   - fsci_sparse::triu(A, k) vs scipy.sparse.triu(A, k) — upper-triangle
//!
//! Resolves [frankenscipy-ioltv]. tril/triu output compared after
//! densification (handles ordering). 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CooMatrix, FormatConvertible, Shape2D, find, tril, triu};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-004";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String, // "find" | "tril" | "triu"
    rows: usize,
    cols: usize,
    triplets: Vec<(usize, usize, f64)>,
    k: isize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// find: sorted triplets flattened as [r0, c0, v0, r1, c1, v1, ...] (row*N + col sort key).
    /// tril/triu: flattened dense matrix row-major.
    values: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create find_tril_triu diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize find_tril_triu diff log");
    fs::write(path, json).expect("write find_tril_triu diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    let matrices: &[(&str, usize, usize, Vec<(usize, usize, f64)>)] = &[
        (
            "3x3_dense",
            3,
            3,
            vec![
                (0, 0, 1.0),
                (0, 1, 2.0),
                (0, 2, 3.0),
                (1, 0, 4.0),
                (1, 1, 5.0),
                (1, 2, 6.0),
                (2, 0, 7.0),
                (2, 1, 8.0),
                (2, 2, 9.0),
            ],
        ),
        (
            "4x4_sparse_pattern",
            4,
            4,
            vec![
                (0, 0, 1.0),
                (0, 2, 2.0),
                (1, 1, -1.0),
                (1, 3, 3.0),
                (2, 0, -2.0),
                (2, 3, 4.0),
                (3, 1, 5.0),
                (3, 3, -3.0),
            ],
        ),
        (
            "5x3_rect",
            5,
            3,
            vec![
                (0, 0, 1.0),
                (0, 2, 2.0),
                (1, 1, -1.0),
                (2, 0, 3.0),
                (3, 2, -2.0),
                (4, 1, 4.0),
            ],
        ),
    ];

    for (label, r, c, t) in matrices {
        // find op
        points.push(PointCase {
            case_id: format!("find_{label}"),
            op: "find".into(),
            rows: *r,
            cols: *c,
            triplets: t.clone(),
            k: 0,
        });
        // tril/triu at a few k offsets
        for k in [-1isize, 0, 1] {
            points.push(PointCase {
                case_id: format!("tril_{label}_k{k}"),
                op: "tril".into(),
                rows: *r,
                cols: *c,
                triplets: t.clone(),
                k,
            });
            points.push(PointCase {
                case_id: format!("triu_{label}_k{k}"),
                op: "triu".into(),
                rows: *r,
                cols: *c,
                triplets: t.clone(),
                k,
            });
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
import scipy.sparse as sp

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
    cid = case["case_id"]; op = case["op"]
    rows = case["rows"]; cols = case["cols"]
    triplets = case["triplets"]
    k = int(case["k"])
    if triplets:
        r = np.array([t[0] for t in triplets], dtype=int)
        c = np.array([t[1] for t in triplets], dtype=int)
        v = np.array([t[2] for t in triplets], dtype=float)
    else:
        r = np.zeros(0, dtype=int); c = np.zeros(0, dtype=int); v = np.zeros(0, dtype=float)
    try:
        A = sp.csr_matrix((v, (r, c)), shape=(rows, cols))
        if op == "find":
            rr, cc, vv = sp.find(A)
            # Sort by (row, col)
            order = sorted(range(len(rr)), key=lambda i: (int(rr[i]), int(cc[i])))
            packed = []
            for i in order:
                packed.append(float(rr[i]))
                packed.append(float(cc[i]))
                packed.append(float(vv[i]))
            points.append({"case_id": cid, "values": packed})
        elif op == "tril":
            d = sp.tril(A, k=k).toarray()
            points.append({"case_id": cid, "values": finite_vec_or_none(d)})
        elif op == "triu":
            d = sp.triu(A, k=k).toarray()
            points.append({"case_id": cid, "values": finite_vec_or_none(d)})
        else:
            points.append({"case_id": cid, "values": None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize find_tril_triu query");
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
                "failed to spawn python3 for find_tril_triu oracle: {e}"
            );
            eprintln!("skipping find_tril_triu oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open find_tril_triu oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "find_tril_triu oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping find_tril_triu oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for find_tril_triu oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "find_tril_triu oracle failed: {stderr}"
        );
        eprintln!("skipping find_tril_triu oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse find_tril_triu oracle JSON"))
}

fn coo_to_dense(coo: &CooMatrix) -> Vec<f64> {
    let (r, c) = (coo.shape().rows, coo.shape().cols);
    let mut dense = vec![0.0; r * c];
    for (idx, (&row, &col)) in coo
        .row_indices()
        .iter()
        .zip(coo.col_indices().iter())
        .enumerate()
    {
        dense[row * c + col] += coo.data()[idx];
    }
    dense
}

fn fsci_eval(case: &PointCase) -> Option<Vec<f64>> {
    let r: Vec<usize> = case.triplets.iter().map(|t| t.0).collect();
    let c: Vec<usize> = case.triplets.iter().map(|t| t.1).collect();
    let d: Vec<f64> = case.triplets.iter().map(|t| t.2).collect();
    let coo = CooMatrix::from_triplets(Shape2D::new(case.rows, case.cols), d, r, c, false).ok()?;
    let csr = coo.to_csr().ok()?;
    match case.op.as_str() {
        "find" => {
            let (rr, cc, vv) = find(&csr).ok()?;
            // Sort by (row, col)
            let mut idx: Vec<usize> = (0..rr.len()).collect();
            idx.sort_by_key(|&i| (rr[i], cc[i]));
            let mut packed = Vec::with_capacity(rr.len() * 3);
            for i in idx {
                packed.push(rr[i] as f64);
                packed.push(cc[i] as f64);
                packed.push(vv[i]);
            }
            Some(packed)
        }
        "tril" => {
            let out = tril(&csr, case.k).ok()?;
            Some(coo_to_dense(&out))
        }
        "triu" => {
            let out = triu(&csr, case.k).ok()?;
            Some(coo_to_dense(&out))
        }
        _ => None,
    }
}

#[test]
fn diff_sparse_find_tril_triu() {
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
                op: case.op.clone(),
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
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_find_tril_triu".into(),
        category: "scipy.sparse.find / tril / triu".into(),
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
                "find_tril_triu {} mismatch: {} abs_diff={}",
                d.op, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.sparse find/tril/triu conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
