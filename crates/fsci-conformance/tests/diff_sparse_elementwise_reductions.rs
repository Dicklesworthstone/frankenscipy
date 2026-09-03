#![forbid(unsafe_code)]
//! Live numpy differential coverage for fsci_sparse elementwise +
//! reduction helpers: sparse_abs, sparse_power, sparse_sum,
//! sparse_row_sums, sparse_col_sums.
//!
//! Resolves [frankenscipy-xgtlt]. 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{
    CsrMatrix, Shape2D, sparse_abs, sparse_col_sums, sparse_power, sparse_row_sums, sparse_sum,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    rows: usize,
    cols: usize,
    dense: Vec<f64>,
    /// Only used by sparse_power.
    p: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    abs: Vec<PointCase>,
    power: Vec<PointCase>,
    sums: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct DenseArm {
    case_id: String,
    dense: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct SumsArm {
    case_id: String,
    total: Option<f64>,
    row_sums: Option<Vec<f64>>,
    col_sums: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    abs: Vec<DenseArm>,
    power: Vec<DenseArm>,
    sums: Vec<SumsArm>,
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
    fs::create_dir_all(output_dir()).expect("create elementwise diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize elementwise diff log");
    fs::write(path, json).expect("write elementwise diff log");
}

fn dense_to_csr(rows: usize, cols: usize, dense: &[f64]) -> CsrMatrix {
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(rows + 1);
    indptr.push(0);
    for r in 0..rows {
        for c in 0..cols {
            let v = dense[r * cols + c];
            if v != 0.0 {
                data.push(v);
                indices.push(c);
            }
        }
        indptr.push(data.len());
    }
    CsrMatrix::from_components(Shape2D::new(rows, cols), data, indices, indptr, true)
        .expect("dense_to_csr build")
}

fn dense_from_csr(csr: &CsrMatrix) -> Vec<f64> {
    let shape = csr.shape();
    let mut dense = vec![0.0_f64; shape.rows * shape.cols];
    let indptr = csr.indptr();
    let indices = csr.indices();
    let data = csr.data();
    for row in 0..shape.rows {
        for idx in indptr[row]..indptr[row + 1] {
            dense[row * shape.cols + indices[idx]] += data[idx];
        }
    }
    dense
}

fn generate_query() -> OracleQuery {
    let mat_3x3 = vec![1.0, 0.0, -2.0, 0.0, 3.0, 0.0, -4.0, 0.0, 5.0];
    let mat_4x5 = vec![
        1.0, 0.0, 0.0, 0.0, 0.5, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.7, 0.0, 0.0, 0.0,
        -4.0, 0.0,
    ];
    let mat_2x4 = vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0];

    let abs_cases = vec![
        PointCase {
            case_id: "abs_3x3_mixed".into(),
            rows: 3,
            cols: 3,
            dense: mat_3x3.clone(),
            p: 0.0,
        },
        PointCase {
            case_id: "abs_4x5".into(),
            rows: 4,
            cols: 5,
            dense: mat_4x5.clone(),
            p: 0.0,
        },
        PointCase {
            case_id: "abs_2x4".into(),
            rows: 2,
            cols: 4,
            dense: mat_2x4.clone(),
            p: 0.0,
        },
    ];

    let mat_3x3_pos = vec![1.0, 0.0, 4.0, 0.0, 9.0, 0.0, 16.0, 0.0, 25.0];
    let power_cases = vec![
        PointCase {
            case_id: "power_3x3_p2".into(),
            rows: 3,
            cols: 3,
            dense: mat_3x3_pos.clone(),
            p: 2.0,
        },
        PointCase {
            case_id: "power_3x3_p0.5".into(),
            rows: 3,
            cols: 3,
            dense: mat_3x3_pos.clone(),
            p: 0.5,
        },
        PointCase {
            case_id: "power_3x3_p3".into(),
            rows: 3,
            cols: 3,
            dense: mat_3x3_pos,
            p: 3.0,
        },
    ];

    let sums_cases = vec![
        PointCase {
            case_id: "sums_3x3".into(),
            rows: 3,
            cols: 3,
            dense: mat_3x3,
            p: 0.0,
        },
        PointCase {
            case_id: "sums_4x5".into(),
            rows: 4,
            cols: 5,
            dense: mat_4x5,
            p: 0.0,
        },
        PointCase {
            case_id: "sums_2x4".into(),
            rows: 2,
            cols: 4,
            dense: mat_2x4,
            p: 0.0,
        },
    ];

    OracleQuery {
        abs: abs_cases,
        power: power_cases,
        sums: sums_cases,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np

def finite_or_none(arr):
    flat = []
    for v in np.asarray(arr, dtype=float).flatten().tolist():
        if not math.isfinite(float(v)):
            return None
        flat.append(float(v))
    return flat

q = json.load(sys.stdin)

abs_out = []
for c in q["abs"]:
    r = int(c["rows"]); cc = int(c["cols"])
    A = np.array(c["dense"], dtype=float).reshape(r, cc)
    abs_A = np.abs(A)
    abs_out.append({"case_id": c["case_id"], "dense": finite_or_none(abs_A)})

power_out = []
for c in q["power"]:
    r = int(c["rows"]); cc = int(c["cols"])
    A = np.array(c["dense"], dtype=float).reshape(r, cc)
    p = float(c["p"])
    # Match fsci semantics: power applied only to stored nonzeros
    # (zero^p stays zero). np.power preserves zeros for positive p,
    # so this is consistent.
    pow_A = np.where(A != 0, np.power(np.abs(A), p) * np.sign(A) ** 0 if False else A, 0.0)
    pow_A = np.where(A != 0, np.power(A, p), 0.0)
    power_out.append({"case_id": c["case_id"], "dense": finite_or_none(pow_A)})

sums_out = []
for c in q["sums"]:
    r = int(c["rows"]); cc = int(c["cols"])
    A = np.array(c["dense"], dtype=float).reshape(r, cc)
    total = float(np.sum(A))
    rs = [float(v) for v in np.sum(A, axis=1).tolist()]
    cs = [float(v) for v in np.sum(A, axis=0).tolist()]
    sums_out.append({"case_id": c["case_id"], "total": total,
                     "row_sums": rs, "col_sums": cs})

print(json.dumps({"abs": abs_out, "power": power_out, "sums": sums_out}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize elementwise query");
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
                "failed to spawn python3 for elementwise oracle: {e}"
            );
            eprintln!("skipping elementwise oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open elementwise oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "elementwise oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping elementwise oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for elementwise oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "elementwise oracle failed: {stderr}"
        );
        eprintln!("skipping elementwise oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse elementwise oracle JSON"))
}

#[test]
fn diff_sparse_elementwise_reductions() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.abs.len(), query.abs.len());
    assert_eq!(oracle.power.len(), query.power.len());
    assert_eq!(oracle.sums.len(), query.sums.len());

    let abs_map: HashMap<String, DenseArm> = oracle
        .abs
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let power_map: HashMap<String, DenseArm> = oracle
        .power
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let sums_map: HashMap<String, SumsArm> = oracle
        .sums
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.abs {
        let scipy_arm = abs_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.dense.as_ref() else {
            continue;
        };
        let csr = dense_to_csr(case.rows, case.cols, &case.dense);
        let out = sparse_abs(&csr);
        let flat = dense_from_csr(&out);
        let abs_d = flat
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "sparse_abs".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    for case in &query.power {
        let scipy_arm = power_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.dense.as_ref() else {
            continue;
        };
        let csr = dense_to_csr(case.rows, case.cols, &case.dense);
        let out = sparse_power(&csr, case.p);
        let flat = dense_from_csr(&out);
        let abs_d = flat
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "sparse_power".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    for case in &query.sums {
        let scipy_arm = sums_map.get(&case.case_id).expect("validated oracle");
        let (Some(total_exp), Some(rs_exp), Some(cs_exp)) = (
            scipy_arm.total,
            scipy_arm.row_sums.as_ref(),
            scipy_arm.col_sums.as_ref(),
        ) else {
            continue;
        };
        let csr = dense_to_csr(case.rows, case.cols, &case.dense);
        let total = sparse_sum(&csr);
        let rs = sparse_row_sums(&csr);
        let cs = sparse_col_sums(&csr);
        let dt = (total - total_exp).abs();
        let dr = rs
            .iter()
            .zip(rs_exp.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let dc = cs
            .iter()
            .zip(cs_exp.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let abs_d = dt.max(dr).max(dc);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "sums".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_elementwise_reductions".into(),
        category: "fsci_sparse abs/power/sums vs numpy".into(),
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
        "elementwise conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
