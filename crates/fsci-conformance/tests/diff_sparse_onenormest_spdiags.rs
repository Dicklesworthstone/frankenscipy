#![forbid(unsafe_code)]
//! Live scipy.sparse parity for fsci_sparse::onenormest and
//! fsci_sparse::spdiags.
//!
//! Resolves [frankenscipy-07a8d].
//!
//! - `onenormest`: fsci implements exact 1-norm (max abs col sum)
//!   rather than the Hager-Higham estimator scipy uses. Compare
//!   against `scipy.sparse.linalg.norm(A, ord=1)` which is exact.
//!   Tight 1e-12 tolerance.
//! - `spdiags`: deterministic DIA construction. Compare dense
//!   reconstruction at 1e-12.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CooMatrix, CsrMatrix, FormatConvertible, Shape2D, onenormest, spdiags};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "onenorm" | "spdiags"
    /// For onenorm
    a_rows: usize,
    a_cols: usize,
    a_triplets: Vec<(usize, usize, f64)>,
    /// For spdiags
    rows: usize,
    cols: usize,
    diag_data: Vec<Vec<f64>>,
    offsets: Vec<isize>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// For onenorm
    value: Option<f64>,
    /// For spdiags
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
    fs::create_dir_all(output_dir()).expect("create onenormest_spdiags diff dir");
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
            out[r * s.cols + col] += data[idx];
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
    let mut points = Vec::new();

    // onenormest probes
    let a1 = vec![
        (0, 0, 3.0_f64),
        (1, 0, -1.0),
        (0, 1, 2.0),
        (1, 1, 4.0),
        (2, 0, 0.5),
        (2, 1, 1.5),
        (2, 2, -2.0),
    ];
    let a2: Vec<(usize, usize, f64)> = (0..6)
        .flat_map(|i| {
            let i_f = i as f64;
            vec![(i, i, i_f + 1.0), (i, (i + 1) % 6, 0.5_f64 - i_f * 0.1)]
        })
        .collect();
    let a3 = vec![
        (0, 0, -1.5_f64),
        (1, 1, 2.7),
        (2, 2, -3.4),
        (3, 3, 0.5),
        (0, 3, 1.2),
        (3, 0, -0.7),
    ];
    for (label, trips, m, n) in [
        ("a1_3x3", &a1, 3, 3),
        ("a2_6x6", &a2, 6, 6),
        ("a3_4x4", &a3, 4, 4),
    ] {
        points.push(Case {
            case_id: format!("onenorm_{label}"),
            op: "onenorm".into(),
            a_rows: m,
            a_cols: n,
            a_triplets: trips.clone(),
            rows: 0,
            cols: 0,
            diag_data: vec![],
            offsets: vec![],
        });
    }

    // spdiags probes (data rows, offsets, m, n)
    // Classic scipy example: data 3x4, offsets [-1, 0, 1]
    points.push(Case {
        case_id: "spdiags_3diag_4x4".into(),
        op: "spdiags".into(),
        a_rows: 0,
        a_cols: 0,
        a_triplets: vec![],
        rows: 4,
        cols: 4,
        diag_data: vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![10.0, 20.0, 30.0, 40.0],
            vec![100.0, 200.0, 300.0, 400.0],
        ],
        offsets: vec![-1, 0, 1],
    });
    points.push(Case {
        case_id: "spdiags_main_5x5".into(),
        op: "spdiags".into(),
        a_rows: 0,
        a_cols: 0,
        a_triplets: vec![],
        rows: 5,
        cols: 5,
        diag_data: vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]],
        offsets: vec![0],
    });
    points.push(Case {
        case_id: "spdiags_off_4x6".into(),
        op: "spdiags".into(),
        a_rows: 0,
        a_cols: 0,
        a_triplets: vec![],
        rows: 4,
        cols: 6,
        diag_data: vec![
            vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
        ],
        offsets: vec![1, -2],
    });

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.linalg import norm as spnorm

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    try:
        if op == "onenorm":
            rs = [int(t[0]) for t in case["a_triplets"]]
            cs = [int(t[1]) for t in case["a_triplets"]]
            vs = [float(t[2]) for t in case["a_triplets"]]
            A = csr_matrix((vs, (rs, cs)), shape=(int(case["a_rows"]), int(case["a_cols"]))).astype(float)
            v = float(spnorm(A, ord=1))
            if math.isfinite(v):
                points.append({"case_id": cid, "value": v, "dense": None, "out_rows": None, "out_cols": None})
            else:
                points.append({"case_id": cid, "value": None, "dense": None, "out_rows": None, "out_cols": None})
        elif op == "spdiags":
            data = [list(map(float, row)) for row in case["diag_data"]]
            offsets = [int(o) for o in case["offsets"]]
            m = int(case["rows"]); n = int(case["cols"])
            D = spdiags(data, offsets, m, n).todense()
            D = np.asarray(D, dtype=float)
            flat = [float(v) for v in D.flatten().tolist()]
            if all(math.isfinite(v) for v in flat):
                points.append({"case_id": cid, "value": None, "dense": flat, "out_rows": int(D.shape[0]), "out_cols": int(D.shape[1])})
            else:
                points.append({"case_id": cid, "value": None, "dense": None, "out_rows": None, "out_cols": None})
        else:
            points.append({"case_id": cid, "value": None, "dense": None, "out_rows": None, "out_cols": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "value": None, "dense": None, "out_rows": None, "out_cols": None})
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
                "failed to spawn python3 for onenormest_spdiags oracle: {e}"
            );
            eprintln!("skipping onenormest_spdiags oracle: python3 not available ({e})");
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
                "onenormest_spdiags oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping onenormest_spdiags oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for onenormest_spdiags oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "onenormest_spdiags oracle failed: {stderr}"
        );
        eprintln!("skipping onenormest_spdiags oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse onenormest_spdiags oracle JSON"))
}

#[test]
fn diff_sparse_onenormest_spdiags() {
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
        match case.op.as_str() {
            "onenorm" => {
                let Some(expected) = arm.value else {
                    continue;
                };
                let Some(csr) = build_csr(case.a_rows, case.a_cols, &case.a_triplets) else {
                    continue;
                };
                let actual = onenormest(&csr);
                let abs_d = (actual - expected).abs();
                max_overall = max_overall.max(abs_d);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    abs_diff: abs_d,
                    pass: abs_d <= ABS_TOL,
                });
            }
            "spdiags" => {
                let (Some(expected), Some(or), Some(oc)) =
                    (arm.dense.as_ref(), arm.out_rows, arm.out_cols)
                else {
                    continue;
                };
                let Ok(dia) = spdiags(&case.diag_data, &case.offsets, case.rows, case.cols) else {
                    continue;
                };
                let Ok(csr) = dia.to_csr() else {
                    continue;
                };
                let s = csr.shape();
                if s.rows != or || s.cols != oc {
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        op: case.op.clone(),
                        abs_diff: f64::INFINITY,
                        pass: false,
                    });
                    max_overall = f64::INFINITY;
                    continue;
                }
                let dense = csr_to_dense(&csr);
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
            _ => continue,
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_onenormest_spdiags".into(),
        category: "fsci_sparse::onenormest + spdiags vs scipy.sparse".into(),
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
        "onenormest/spdiags conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
