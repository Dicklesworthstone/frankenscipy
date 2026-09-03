#![forbid(unsafe_code)]
//! Live numpy differential coverage for fsci_sparse::sparse_density and
//! sparse_frobenius_inner.
//!
//! Resolves [frankenscipy-pm8kb]. 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CsrMatrix, Shape2D, sparse_density, sparse_frobenius_inner};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct DensityCase {
    case_id: String,
    rows: usize,
    cols: usize,
    dense: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct FrobCase {
    case_id: String,
    rows: usize,
    cols: usize,
    a: Vec<f64>,
    b: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    density: Vec<DensityCase>,
    frob: Vec<FrobCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct ScalarArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    density: Vec<ScalarArm>,
    frob: Vec<ScalarArm>,
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
    fs::create_dir_all(output_dir()).expect("create density_frob diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize density_frob diff log");
    fs::write(path, json).expect("write density_frob diff log");
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

fn generate_query() -> OracleQuery {
    let dense_3x3 = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0];
    let dense_4x5 = vec![
        1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.7, 0.0, 0.0, 0.0,
        4.0, 0.0,
    ];
    let dense_5x5_dense = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0, 4.0, 5.0, 6.0,
        7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    ];

    let density = vec![
        DensityCase {
            case_id: "3x3_diag_off".into(),
            rows: 3,
            cols: 3,
            dense: dense_3x3.clone(),
        },
        DensityCase {
            case_id: "4x5_sparse".into(),
            rows: 4,
            cols: 5,
            dense: dense_4x5.clone(),
        },
        DensityCase {
            case_id: "5x5_full".into(),
            rows: 5,
            cols: 5,
            dense: dense_5x5_dense.clone(),
        },
    ];

    let a_5x5_alt = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0,
    ];
    let b_5x5_alt = vec![
        2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
        2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0,
    ];
    let frob = vec![
        FrobCase {
            case_id: "3x3_self".into(),
            rows: 3,
            cols: 3,
            a: dense_3x3.clone(),
            b: dense_3x3.clone(),
        },
        FrobCase {
            case_id: "4x5_self".into(),
            rows: 4,
            cols: 5,
            a: dense_4x5.clone(),
            b: dense_4x5.clone(),
        },
        FrobCase {
            case_id: "5x5_diag_pair".into(),
            rows: 5,
            cols: 5,
            a: a_5x5_alt,
            b: b_5x5_alt,
        },
        FrobCase {
            case_id: "5x5_full_self".into(),
            rows: 5,
            cols: 5,
            a: dense_5x5_dense.clone(),
            b: dense_5x5_dense,
        },
    ];

    OracleQuery { density, frob }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np

q = json.load(sys.stdin)

density = []
for c in q["density"]:
    cid = c["case_id"]
    r = int(c["rows"]); cc = int(c["cols"])
    A = np.array(c["dense"], dtype=float).reshape(r, cc)
    try:
        nnz = int(np.count_nonzero(A))
        total = r * cc
        v = float(nnz / total) if total > 0 else 0.0
        density.append({"case_id": cid, "value": v})
    except Exception:
        density.append({"case_id": cid, "value": None})

frob = []
for c in q["frob"]:
    cid = c["case_id"]
    r = int(c["rows"]); cc = int(c["cols"])
    A = np.array(c["a"], dtype=float).reshape(r, cc)
    B = np.array(c["b"], dtype=float).reshape(r, cc)
    try:
        v = float(np.sum(A * B))
        if not math.isfinite(v):
            frob.append({"case_id": cid, "value": None})
        else:
            frob.append({"case_id": cid, "value": v})
    except Exception:
        frob.append({"case_id": cid, "value": None})

print(json.dumps({"density": density, "frob": frob}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize density_frob query");
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
                "failed to spawn python3 for density_frob oracle: {e}"
            );
            eprintln!("skipping density_frob oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open density_frob oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "density_frob oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping density_frob oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for density_frob oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "density_frob oracle failed: {stderr}"
        );
        eprintln!("skipping density_frob oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse density_frob oracle JSON"))
}

#[test]
fn diff_sparse_density_frob_inner() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.density.len(), query.density.len());
    assert_eq!(oracle.frob.len(), query.frob.len());

    let density_map: HashMap<String, ScalarArm> = oracle
        .density
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let frob_map: HashMap<String, ScalarArm> = oracle
        .frob
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.density {
        let scipy_arm = density_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.value else {
            continue;
        };
        let csr = dense_to_csr(case.rows, case.cols, &case.dense);
        let fsci_v = sparse_density(&csr);
        let abs_d = (fsci_v - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "density".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    for case in &query.frob {
        let scipy_arm = frob_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.value else {
            continue;
        };
        let a_csr = dense_to_csr(case.rows, case.cols, &case.a);
        let b_csr = dense_to_csr(case.rows, case.cols, &case.b);
        let fsci_v = sparse_frobenius_inner(&a_csr, &b_csr);
        let abs_d = (fsci_v - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "frob_inner".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_density_frob_inner".into(),
        category: "fsci_sparse density + frobenius_inner".into(),
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
        "density_frob_inner conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
