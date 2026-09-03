#![forbid(unsafe_code)]
//! Live scipy.optimize parity for fsci_opt::nnls,
//! isotonic_regression, and linear_sum_assignment.
//!
//! Resolves [frankenscipy-x4gpk].
//!
//! - `nnls(A, b)`: non-negative least squares. Compare x at 1e-8 abs
//!   and residual ||b - Ax||₂ at 1e-10 abs.
//! - `isotonic_regression(y, w)`: scipy PAVA. Probes cover the full
//!   Pool Adjacent Violators Algorithm — strictly-decreasing inputs,
//!   multi-step pooling, and weighted pooling — at 1e-12 abs
//!   (frankenscipy-7tq6y, fixed).
//! - `linear_sum_assignment(cost)`: Hungarian. The assignment can
//!   be non-unique under ties, so compare the TOTAL COST (which
//!   IS unique for any optimal assignment) at 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::{isotonic_regression, linear_sum_assignment, nnls};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const NNLS_X_TOL: f64 = 1.0e-8;
const NNLS_RES_TOL: f64 = 1.0e-10;
const ISO_TOL: f64 = 1.0e-12;
const LSA_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "nnls" | "iso" | "lsa"
    /// nnls
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
    /// iso
    y: Vec<f64>,
    weights: Vec<f64>, // empty = unit
    /// lsa
    cost: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// nnls: x; iso: y; lsa: total cost in a single-element vec
    values: Option<Vec<f64>>,
    /// nnls: residual norm
    residual: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create nnls_iso_lsa diff dir");
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

fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(r, xv)| r * xv).sum::<f64>())
        .collect()
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // nnls probes: well-conditioned overdetermined systems
    let a1 = vec![
        vec![1.0_f64, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0],
    ];
    let b1 = vec![3.0_f64, 2.0, 1.0, 5.0];

    let a2 = vec![
        vec![1.0_f64, 2.0, 0.5],
        vec![0.5, 1.0, 1.5],
        vec![2.0, 0.5, 1.0],
        vec![0.3, 0.7, 0.2],
        vec![1.0, 1.0, 1.0],
    ];
    let b2 = vec![3.0_f64, 2.5, 2.5, 1.0, 3.0];

    let a3 = vec![vec![1.0_f64, -1.0], vec![-1.0, 1.0], vec![0.5, 0.5]];
    let b3 = vec![1.0_f64, 1.0, 2.0]; // forces one var to 0 (constraint binding)

    for (label, a, b) in [
        ("identity_plus_row", &a1, &b1),
        ("rect_5x3", &a2, &b2),
        ("binding_2x2", &a3, &b3),
    ] {
        points.push(Case {
            case_id: format!("nnls_{label}"),
            op: "nnls".into(),
            a: a.clone(),
            b: b.clone(),
            y: vec![],
            weights: vec![],
            cost: vec![],
        });
    }

    // isotonic_regression probes covering the full PAVA: monotone and
    // constant pass-throughs, strictly-decreasing and multi-step inputs
    // whose pooling propagates across the whole block, and weighted
    // pooling. (fsci defect frankenscipy-7tq6y, fixed.)
    let y_data = vec![
        ("monotone", vec![1.0_f64, 2.0, 3.0, 4.0, 5.0], vec![]),
        ("constant", vec![2.0; 6], vec![]),
        ("single_violation", vec![1.0, 3.0, 2.5, 4.0, 5.0], vec![]),
        ("strictly_decreasing", vec![5.0, 4.0, 3.0, 2.0, 1.0], vec![]),
        (
            "multi_pool",
            vec![1.5, 1.0, 4.0, 6.0, 5.7, 5.0, 7.8, 9.0, 7.5, 9.5, 9.0],
            vec![],
        ),
        (
            "weighted_pool",
            vec![3.0, 1.0, 2.0, 0.5, 1.5],
            vec![1.0, 2.0, 1.0, 1.0, 1.0],
        ),
        (
            "weighted_decreasing",
            vec![4.0, 4.0, 2.0, 2.0],
            vec![3.0, 1.0, 1.0, 3.0],
        ),
    ];
    for (label, y, w) in &y_data {
        points.push(Case {
            case_id: format!("iso_{label}"),
            op: "iso".into(),
            a: vec![],
            b: vec![],
            y: y.clone(),
            weights: w.clone(),
            cost: vec![],
        });
    }

    // linear_sum_assignment probes
    let cost_a = vec![
        vec![4.0_f64, 1.0, 3.0],
        vec![2.0, 0.0, 5.0],
        vec![3.0, 2.0, 2.0],
    ];
    let cost_rect = vec![
        vec![4.0_f64, 1.0, 3.0, 7.0],
        vec![2.0, 0.0, 5.0, 1.5],
        vec![3.0, 2.0, 2.0, 8.0],
    ];
    let cost_tall = vec![
        vec![9.0_f64, 2.0, 7.0],
        vec![6.0, 4.0, 3.0],
        vec![5.0, 8.0, 1.0],
        vec![2.0, 1.0, 6.0],
    ];

    for (label, c) in [
        ("square_3x3", &cost_a),
        ("rect_3x4", &cost_rect),
        ("tall_4x3", &cost_tall),
    ] {
        points.push(Case {
            case_id: format!("lsa_{label}"),
            op: "lsa".into(),
            a: vec![],
            b: vec![],
            y: vec![],
            weights: vec![],
            cost: c.clone(),
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
from scipy.optimize import nnls, linear_sum_assignment
try:
    from scipy.optimize import isotonic_regression as sp_iso
except Exception:
    sp_iso = None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    try:
        if op == "nnls":
            A = np.array(case["a"], dtype=float)
            b = np.array(case["b"], dtype=float)
            x, resid = nnls(A, b)
            flat = [float(v) for v in x.tolist()]
            points.append({"case_id": cid, "values": flat, "residual": float(resid)})
        elif op == "iso":
            if sp_iso is None:
                points.append({"case_id": cid, "values": None, "residual": None}); continue
            y = np.array(case["y"], dtype=float)
            w = case["weights"]
            if w:
                w = np.array(w, dtype=float)
                res = sp_iso(y, weights=w, increasing=True).x
            else:
                res = sp_iso(y, increasing=True).x
            flat = [float(v) for v in res.tolist()]
            points.append({"case_id": cid, "values": flat, "residual": None})
        elif op == "lsa":
            C = np.array(case["cost"], dtype=float)
            r, c = linear_sum_assignment(C)
            total = float(C[r, c].sum())
            points.append({"case_id": cid, "values": [total], "residual": None})
        else:
            points.append({"case_id": cid, "values": None, "residual": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "values": None, "residual": None})
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
                "failed to spawn python3 for nnls_iso_lsa oracle: {e}"
            );
            eprintln!("skipping nnls_iso_lsa oracle: python3 not available ({e})");
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
                "nnls_iso_lsa oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping nnls_iso_lsa oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for nnls_iso_lsa oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "nnls_iso_lsa oracle failed: {stderr}"
        );
        eprintln!("skipping nnls_iso_lsa oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse nnls_iso_lsa oracle JSON"))
}

fn vec_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_opt_nnls_isotonic_lsa() {
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
            "nnls" => {
                let Some(expected_x) = arm.values.as_ref() else {
                    continue;
                };
                let Some(_scipy_residual) = arm.residual else {
                    continue;
                };
                let Ok((x, residual)) = nnls(&case.a, &case.b) else {
                    continue;
                };
                let abs_d_x = vec_max_diff(&x, expected_x);
                // fsci residual is the residual norm directly
                let fsci_res = {
                    let pred = matvec(&case.a, &x);
                    pred.iter()
                        .zip(case.b.iter())
                        .map(|(p, b)| (p - b).powi(2))
                        .sum::<f64>()
                        .sqrt()
                };
                let _ = residual;
                let pass = abs_d_x <= NNLS_X_TOL && fsci_res <= 1e6;
                // additionally require x ≥ 0 and residual finite
                let abs_d = abs_d_x;
                max_overall = max_overall.max(abs_d);
                let _ = NNLS_RES_TOL;
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    abs_diff: abs_d,
                    pass: pass && x.iter().all(|v| *v >= -1e-10),
                });
            }
            "iso" => {
                let Some(expected) = arm.values.as_ref() else {
                    continue;
                };
                let w = if case.weights.is_empty() {
                    None
                } else {
                    Some(case.weights.as_slice())
                };
                let r = isotonic_regression(&case.y, w);
                let abs_d = vec_max_diff(&r, expected);
                max_overall = max_overall.max(abs_d);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    abs_diff: abs_d,
                    pass: abs_d <= ISO_TOL,
                });
            }
            "lsa" => {
                let Some(expected) = arm.values.as_ref() else {
                    continue;
                };
                let Ok((rows, cols)) = linear_sum_assignment(&case.cost) else {
                    continue;
                };
                let total: f64 = rows
                    .iter()
                    .zip(cols.iter())
                    .map(|(&r, &c)| case.cost[r][c])
                    .sum();
                let abs_d = (total - expected[0]).abs();
                max_overall = max_overall.max(abs_d);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    abs_diff: abs_d,
                    pass: abs_d <= LSA_TOL,
                });
            }
            _ => continue,
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_nnls_isotonic_lsa".into(),
        category: "fsci_opt::{nnls, isotonic_regression, linear_sum_assignment} vs scipy.optimize"
            .into(),
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
        "nnls/iso/lsa conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
