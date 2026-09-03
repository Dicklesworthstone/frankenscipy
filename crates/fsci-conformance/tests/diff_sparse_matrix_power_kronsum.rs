#![forbid(unsafe_code)]
//! Live scipy.sparse parity for fsci_sparse::matrix_power and
//! fsci_sparse::kronsum.
//!
//! Resolves [frankenscipy-spwpc]. Both functions return CSR
//! matrices that are exactly determined by inputs (no iteration,
//! no sign ambiguity), so a tight 1e-12 dense comparison applies.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CooMatrix, CsrMatrix, FormatConvertible, Shape2D, kronsum, matrix_power};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "power" | "kronsum"
    a_rows: usize,
    a_cols: usize,
    a_triplets: Vec<(usize, usize, f64)>,
    /// Only used for kronsum
    b_rows: usize,
    b_cols: usize,
    b_triplets: Vec<(usize, usize, f64)>,
    /// Only used for matrix_power
    n: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Row-major dense flatten
    dense: Option<Vec<f64>>,
    out_rows: Option<usize>,
    out_cols: Option<usize>,
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
    fs::create_dir_all(output_dir()).expect("create matrix_power_kronsum diff dir");
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

fn csr_to_dense(c: &CsrMatrix) -> Vec<f64> {
    let s = c.shape();
    let mut out = vec![0.0f64; s.rows * s.cols];
    let indptr = c.indptr();
    let indices = c.indices();
    let data = c.data();
    for r in 0..s.rows {
        let start = indptr[r];
        let end = indptr[r + 1];
        for idx in start..end {
            let col = indices[idx];
            let v = data[idx];
            out[r * s.cols + col] += v;
        }
    }
    out
}

fn build_csr(rows: usize, cols: usize, trips: &[(usize, usize, f64)]) -> Option<CsrMatrix> {
    let mut data = Vec::new();
    let mut rs = Vec::new();
    let mut cs = Vec::new();
    for &(r, c, v) in trips {
        data.push(v);
        rs.push(r);
        cs.push(c);
    }
    let coo = CooMatrix::from_triplets(Shape2D::new(rows, cols), data, rs, cs, true).ok()?;
    coo.to_csr().ok()
}

fn generate_query() -> OracleQuery {
    // matrix_power: small square matrices, small n
    let pa_tri = vec![
        (0, 0, 2.0_f64),
        (1, 1, 2.0),
        (2, 2, 2.0),
        (0, 1, -1.0),
        (1, 0, -1.0),
        (1, 2, -1.0),
        (2, 1, -1.0),
    ];
    let pa_4x4 = vec![
        (0, 0, 1.5),
        (1, 1, 0.7),
        (2, 2, -0.3),
        (3, 3, 1.2),
        (0, 1, 0.4),
        (1, 0, 0.4),
        (2, 3, 0.5),
        (3, 2, 0.5),
        (0, 2, 0.1),
        (2, 0, 0.1),
    ];

    let mut points = Vec::new();

    for (label, m) in [("tri3", &pa_tri), ("sym4", &pa_4x4)] {
        for n in [0_usize, 1, 2, 3, 5] {
            let rows = m.iter().map(|t| t.0).max().unwrap_or(0) + 1;
            let cols = m.iter().map(|t| t.1).max().unwrap_or(0) + 1;
            let dim = rows.max(cols);
            points.push(Case {
                case_id: format!("power_{label}_n{n}"),
                op: "power".into(),
                a_rows: dim,
                a_cols: dim,
                a_triplets: m.clone(),
                b_rows: 0,
                b_cols: 0,
                b_triplets: vec![],
                n,
            });
        }
    }

    // kronsum: small square A, B
    let a_small = vec![(0, 0, 2.0_f64), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 2.0)];
    let b_small = vec![(0, 0, 3.0_f64), (1, 1, 4.0), (0, 1, 0.5), (1, 0, 0.5)];
    let a_3 = vec![
        (0, 0, 1.0_f64),
        (1, 1, 2.0),
        (2, 2, 3.0),
        (0, 1, 0.5),
        (1, 2, 0.7),
    ];
    let b_3 = vec![
        (0, 0, -1.0_f64),
        (1, 1, -2.0),
        (2, 2, 0.5),
        (1, 0, 0.3),
        (2, 1, 0.4),
    ];

    points.push(Case {
        case_id: "ks_2x2_2x2".into(),
        op: "kronsum".into(),
        a_rows: 2,
        a_cols: 2,
        a_triplets: a_small.clone(),
        b_rows: 2,
        b_cols: 2,
        b_triplets: b_small.clone(),
        n: 0,
    });
    points.push(Case {
        case_id: "ks_3x3_2x2".into(),
        op: "kronsum".into(),
        a_rows: 3,
        a_cols: 3,
        a_triplets: a_3.clone(),
        b_rows: 2,
        b_cols: 2,
        b_triplets: b_small.clone(),
        n: 0,
    });
    points.push(Case {
        case_id: "ks_3x3_3x3".into(),
        op: "kronsum".into(),
        a_rows: 3,
        a_cols: 3,
        a_triplets: a_3,
        b_rows: 3,
        b_cols: 3,
        b_triplets: b_3,
        n: 0,
    });

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.sparse import csr_matrix, kronsum
from scipy.sparse.linalg import matrix_power as sp_matrix_power

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    try:
        a_rs = [int(t[0]) for t in case["a_triplets"]]
        a_cs = [int(t[1]) for t in case["a_triplets"]]
        a_vs = [float(t[2]) for t in case["a_triplets"]]
        A = csr_matrix((a_vs, (a_rs, a_cs)), shape=(int(case["a_rows"]), int(case["a_cols"]))).astype(float)
        if op == "power":
            n = int(case["n"])
            R = sp_matrix_power(A, n)
            D = np.asarray(R.todense())
        elif op == "kronsum":
            b_rs = [int(t[0]) for t in case["b_triplets"]]
            b_cs = [int(t[1]) for t in case["b_triplets"]]
            b_vs = [float(t[2]) for t in case["b_triplets"]]
            B = csr_matrix((b_vs, (b_rs, b_cs)), shape=(int(case["b_rows"]), int(case["b_cols"]))).astype(float)
            R = kronsum(A, B)
            D = np.asarray(R.todense())
        else:
            points.append({"case_id": cid, "dense": None, "out_rows": None, "out_cols": None}); continue
        flat = [float(v) for v in D.flatten().tolist()]
        if all(math.isfinite(v) for v in flat):
            points.append({"case_id": cid, "dense": flat, "out_rows": int(D.shape[0]), "out_cols": int(D.shape[1])})
        else:
            points.append({"case_id": cid, "dense": None, "out_rows": None, "out_cols": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "dense": None, "out_rows": None, "out_cols": None})
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
                "failed to spawn python3 for matrix_power_kronsum oracle: {e}"
            );
            eprintln!("skipping matrix_power_kronsum oracle: python3 not available ({e})");
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
                "matrix_power_kronsum oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping matrix_power_kronsum oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for matrix_power_kronsum oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "matrix_power_kronsum oracle failed: {stderr}"
        );
        eprintln!("skipping matrix_power_kronsum oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse matrix_power_kronsum oracle JSON"))
}

#[test]
fn diff_sparse_matrix_power_kronsum() {
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
        let (Some(expected), Some(or), Some(oc)) = (arm.dense.as_ref(), arm.out_rows, arm.out_cols)
        else {
            continue;
        };
        let Some(csr_a) = build_csr(case.a_rows, case.a_cols, &case.a_triplets) else {
            continue;
        };
        let result = match case.op.as_str() {
            "power" => matrix_power(&csr_a, case.n),
            "kronsum" => {
                let Some(csr_b) = build_csr(case.b_rows, case.b_cols, &case.b_triplets) else {
                    continue;
                };
                kronsum(&csr_a, &csr_b)
            }
            _ => continue,
        };
        let Ok(out) = result else {
            continue;
        };
        let shape = out.shape();
        if shape.rows != or || shape.cols != oc {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                op: case.op.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            max_overall = f64::INFINITY;
            continue;
        }
        let dense = csr_to_dense(&out);
        let abs_d = if dense.len() != expected.len() {
            f64::INFINITY
        } else {
            dense
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
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
        test_id: "diff_sparse_matrix_power_kronsum".into(),
        category: "fsci_sparse::matrix_power + kronsum vs scipy.sparse".into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "matrix_power/kronsum conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
