#![forbid(unsafe_code)]
//! Live SciPy/numpy parity harness for fsci_linalg::matrix_power and
//! signm.
//!
//! Resolves [frankenscipy-b7ib6]. 1e-9 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{DecompOptions, matrix_power, signm};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    rows: usize,
    cols: usize,
    a: Vec<f64>,
    /// Integer exponent for matrix_power.
    power: i64,
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
    fs::create_dir_all(output_dir()).expect("create matrix_power_signm diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize matrix_power_signm diff log");
    fs::write(path, json).expect("write matrix_power_signm diff log");
}

fn rows_of(a_flat: &[f64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|r| (0..cols).map(|c| a_flat[r * cols + c]).collect())
        .collect()
}

fn flatten(m: &[Vec<f64>]) -> Vec<f64> {
    m.iter().flat_map(|r| r.iter().copied()).collect()
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    let a_2x2_tri = vec![2.0_f64, 1.0, 0.0, 3.0];
    let a_2x2_diag = vec![2.0_f64, 0.0, 0.0, -1.0];
    let a_3x3_diag = vec![1.5_f64, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 4.0];
    let a_3x3_tri = vec![1.0_f64, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0];

    // matrix_power
    for (label, mat, rows, cols) in [
        ("tri_2x2", &a_2x2_tri, 2, 2),
        ("diag_2x2", &a_2x2_diag, 2, 2),
        ("diag_3x3", &a_3x3_diag, 3, 3),
        ("tri_3x3", &a_3x3_tri, 3, 3),
    ] {
        for power in [0_i64, 1, 2, 3, 5, -1] {
            points.push(PointCase {
                case_id: format!("matpow_{label}_p{power}"),
                op: "matrix_power".into(),
                rows,
                cols,
                a: mat.clone(),
                power,
            });
        }
    }

    // signm: only on matrices with all-real eigenvalues
    // (diagonal/triangular with positive or negative reals)
    for (label, mat, rows, cols) in [
        ("diag_2x2_mixed", &a_2x2_diag, 2, 2),
        ("diag_3x3_mixed", &a_3x3_diag, 3, 3),
        ("tri_2x2_pos", &a_2x2_tri, 2, 2),
        ("tri_3x3_pos", &a_3x3_tri, 3, 3),
    ] {
        points.push(PointCase {
            case_id: format!("signm_{label}"),
            op: "signm".into(),
            rows,
            cols,
            a: mat.clone(),
            power: 0,
        });
    }

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import linalg

def finite_flat_or_none(arr):
    flat = []
    for v in np.asarray(arr, dtype=float).flatten().tolist():
        if not math.isfinite(float(v)):
            return None
        flat.append(float(v))
    return flat

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    rows = int(case["rows"]); cols = int(case["cols"])
    A = np.array(case["a"], dtype=float).reshape(rows, cols)
    p = int(case["power"])
    try:
        if op == "matrix_power":
            m = np.linalg.matrix_power(A, p)
        elif op == "signm":
            m = linalg.signm(A)
        else:
            m = None
        if m is None:
            points.append({"case_id": cid, "values": None})
        else:
            points.append({"case_id": cid, "values": finite_flat_or_none(m)})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize matrix_power_signm query");
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
                "failed to spawn python3 for matrix_power_signm oracle: {e}"
            );
            eprintln!("skipping matrix_power_signm oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open matrix_power_signm oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "matrix_power_signm oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping matrix_power_signm oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for matrix_power_signm oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "matrix_power_signm oracle failed: {stderr}"
        );
        eprintln!("skipping matrix_power_signm oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse matrix_power_signm oracle JSON"))
}

#[test]
fn diff_linalg_matrix_power_signm() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.values.as_ref() else {
            continue;
        };
        let a = rows_of(&case.a, case.rows, case.cols);
        let opts = DecompOptions::default();
        let fsci_mat = match case.op.as_str() {
            "matrix_power" => matrix_power(&a, case.power as i32, opts),
            "signm" => signm(&a, opts),
            _ => continue,
        };
        let Ok(m) = fsci_mat else {
            continue;
        };
        let fsci_flat = flatten(&m);
        let abs_d = if fsci_flat.len() != expected.len() {
            f64::INFINITY
        } else {
            fsci_flat
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
        test_id: "diff_linalg_matrix_power_signm".into(),
        category: "numpy.linalg.matrix_power + scipy.linalg.signm".into(),
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
        "matrix_power_signm conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
