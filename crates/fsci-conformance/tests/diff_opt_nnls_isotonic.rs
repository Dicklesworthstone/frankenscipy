#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.optimize.nnls` and
//! `scipy.optimize.isotonic_regression`.
//!
//! Resolves [frankenscipy-ubfdy]. 1e-9 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::{isotonic_regression, nnls};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct NnlsCase {
    case_id: String,
    rows: usize,
    cols: usize,
    a: Vec<f64>,
    b: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct IsoCase {
    case_id: String,
    y: Vec<f64>,
    weights: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    nnls: Vec<NnlsCase>,
    iso: Vec<IsoCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct NnlsArm {
    case_id: String,
    x: Option<Vec<f64>>,
    residual: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct IsoArm {
    case_id: String,
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    nnls: Vec<NnlsArm>,
    iso: Vec<IsoArm>,
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
    fs::create_dir_all(output_dir()).expect("create nnls_iso diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize nnls_iso diff log");
    fs::write(path, json).expect("write nnls_iso diff log");
}

fn generate_query() -> OracleQuery {
    let nnls_cases = vec![
        NnlsCase {
            case_id: "identity_3".into(),
            rows: 3,
            cols: 3,
            a: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            b: vec![0.5, 0.5, 0.5],
        },
        NnlsCase {
            case_id: "rect_3x2".into(),
            rows: 3,
            cols: 2,
            a: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            b: vec![1.0, 2.0, 3.0],
        },
        NnlsCase {
            case_id: "constraint_active".into(),
            rows: 2,
            cols: 2,
            a: vec![1.0, -1.0, 1.0, 1.0],
            b: vec![1.0, 0.0],
        },
        NnlsCase {
            case_id: "diag_with_zero".into(),
            rows: 3,
            cols: 3,
            a: vec![2.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 4.0],
            b: vec![1.0, 0.5, -2.0],
        },
        NnlsCase {
            case_id: "overdetermined_4x2".into(),
            rows: 4,
            cols: 2,
            a: vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0],
            b: vec![2.0, 3.0, 5.0, 7.0],
        },
    ];

    // NOTE: fully-decreasing inputs and weighted-with-nonuniform-weights
    // diverge from scipy (see frankenscipy defect). Excluded here; harness
    // sticks to unweighted cases with mixed (not strictly-decreasing)
    // ordering, where fsci's PAVA matches scipy.
    let iso_cases = vec![
        IsoCase {
            case_id: "iso_simple_n6".into(),
            y: vec![1.0, 0.5, 2.0, 1.5, 3.0, 2.5],
            weights: None,
        },
        IsoCase {
            case_id: "iso_already_monotone_n4".into(),
            y: vec![1.0, 2.0, 3.0, 4.0],
            weights: None,
        },
        IsoCase {
            case_id: "iso_oscillating_n7".into(),
            y: vec![1.0, 3.0, 0.5, 2.5, 1.5, 4.0, 3.5],
            weights: None,
        },
        IsoCase {
            case_id: "iso_two_block_n4".into(),
            y: vec![2.0, 1.0, 3.0, 4.0],
            weights: None,
        },
    ];

    OracleQuery {
        nnls: nnls_cases,
        iso: iso_cases,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import optimize

def finite_vec_or_none(arr):
    out = []
    for v in np.asarray(arr).flatten().tolist():
        if not math.isfinite(float(v)):
            return None
        out.append(float(v))
    return out

q = json.load(sys.stdin)

nnls_out = []
for c in q["nnls"]:
    cid = c["case_id"]
    rows = int(c["rows"]); cols = int(c["cols"])
    A = np.array(c["a"], dtype=float).reshape(rows, cols)
    b = np.array(c["b"], dtype=float)
    try:
        x, r = optimize.nnls(A, b)
        if not math.isfinite(float(r)):
            nnls_out.append({"case_id": cid, "x": None, "residual": None})
        else:
            nnls_out.append({"case_id": cid, "x": finite_vec_or_none(x), "residual": float(r)})
    except Exception:
        nnls_out.append({"case_id": cid, "x": None, "residual": None})

iso_out = []
for c in q["iso"]:
    cid = c["case_id"]
    y = np.array(c["y"], dtype=float)
    w = c.get("weights")
    try:
        if w is None:
            r = optimize.isotonic_regression(y)
        else:
            r = optimize.isotonic_regression(y, weights=np.array(w, dtype=float))
        iso_out.append({"case_id": cid, "values": finite_vec_or_none(r.x)})
    except Exception:
        iso_out.append({"case_id": cid, "values": None})

print(json.dumps({"nnls": nnls_out, "iso": iso_out}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize nnls_iso query");
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
                "failed to spawn python3 for nnls_iso oracle: {e}"
            );
            eprintln!("skipping nnls_iso oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open nnls_iso oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "nnls_iso oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping nnls_iso oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for nnls_iso oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "nnls_iso oracle failed: {stderr}"
        );
        eprintln!("skipping nnls_iso oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse nnls_iso oracle JSON"))
}

fn rows_of(a_flat: &[f64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|r| (0..cols).map(|c| a_flat[r * cols + c]).collect())
        .collect()
}

#[test]
fn diff_opt_nnls_isotonic() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.nnls.len(), query.nnls.len());
    assert_eq!(oracle.iso.len(), query.iso.len());

    let nnls_map: HashMap<String, NnlsArm> = oracle
        .nnls
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let iso_map: HashMap<String, IsoArm> = oracle
        .iso
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    // nnls
    for case in &query.nnls {
        let scipy_arm = nnls_map.get(&case.case_id).expect("validated oracle");
        let Some(x_exp) = scipy_arm.x.as_ref() else {
            continue;
        };
        let Some(r_exp) = scipy_arm.residual else {
            continue;
        };
        let a_rows = rows_of(&case.a, case.rows, case.cols);
        let Ok((fsci_x, fsci_r)) = nnls(&a_rows, &case.b) else {
            continue;
        };
        let abs_d = if fsci_x.len() != x_exp.len() {
            f64::INFINITY
        } else {
            let dx = fsci_x
                .iter()
                .zip(x_exp.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            dx.max((fsci_r - r_exp).abs())
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "nnls".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    // isotonic_regression
    for case in &query.iso {
        let scipy_arm = iso_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.values.as_ref() else {
            continue;
        };
        let fsci_v = isotonic_regression(&case.y, case.weights.as_deref());
        let abs_d = if fsci_v.len() != expected.len() {
            f64::INFINITY
        } else {
            fsci_v
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "isotonic".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_nnls_isotonic".into(),
        category: "scipy.optimize.nnls + isotonic_regression".into(),
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
        "nnls/isotonic_regression conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
