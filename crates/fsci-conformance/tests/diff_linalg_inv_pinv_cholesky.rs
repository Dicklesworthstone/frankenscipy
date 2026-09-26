#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_linalg matrix
//! inversion and Cholesky decomposition.
//!
//! Resolves [frankenscipy-dcsnd]. Covers:
//!   • inv      vs scipy.linalg.inv      (square invertible)
//!   • pinv     vs scipy.linalg.pinv     (general, incl. non-square)
//!   • cholesky vs scipy.linalg.cholesky (SPD; lower=True)
//!
//! 4 fixtures × {applicable fns} ≈ 10 cases. Tol 1e-9 abs.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_linalg::{DecompOptions, InvOptions, PinvOptions, cholesky, inv, pinv};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-013";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    a: Vec<Vec<f64>>,
    is_spd: bool,
    is_square_invertible: bool,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    inv: Option<Vec<Vec<f64>>>,
    pinv: Option<Vec<Vec<f64>>>,
    chol_lower: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    sub_check: String,
    max_abs_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
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
    fs::create_dir_all(output_dir()).expect("create inv_pinv_cholesky diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize inv_pinv_cholesky diff log");
    fs::write(path, json).expect("write inv_pinv_cholesky diff log");
}

fn max_abs_diff_mat(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut m = 0.0_f64;
    for (ra, rb) in a.iter().zip(b.iter()) {
        for (va, vb) in ra.iter().zip(rb.iter()) {
            m = m.max((va - vb).abs());
        }
    }
    m
}

/// `slices` over the row-major flattening records a missing side, a length mismatch, or a
/// non-finite entry (which the max fold in `max_abs_diff_mat` would swallow); the metric is
/// still `max_abs_diff_mat`.
fn matrix_diff(
    ledger: &mut CompareLedger,
    arm: &str,
    case_id: &str,
    scipy: Option<&[Vec<f64>]>,
    fsci: Option<&[Vec<f64>]>,
) -> Option<f64> {
    let scipy_flat: Option<Vec<f64>> = scipy.map(<[Vec<f64>]>::concat);
    let fsci_flat: Option<Vec<f64>> = fsci.map(<[Vec<f64>]>::concat);
    ledger.slices(arm, case_id, scipy_flat.as_deref(), fsci_flat.as_deref())?;
    Some(max_abs_diff_mat(fsci?, scipy?))
}

fn generate_query() -> OracleQuery {
    OracleQuery {
        points: vec![
            // 2×2 SPD
            PointCase {
                case_id: "spd_2x2".into(),
                a: vec![vec![4.0, 1.0], vec![1.0, 3.0]],
                is_spd: true,
                is_square_invertible: true,
            },
            // 3×3 SPD
            PointCase {
                case_id: "spd_3x3".into(),
                a: vec![
                    vec![4.0, 1.0, 0.5],
                    vec![1.0, 3.0, 0.25],
                    vec![0.5, 0.25, 2.0],
                ],
                is_spd: true,
                is_square_invertible: true,
            },
            // 3×3 square non-symmetric, invertible
            PointCase {
                case_id: "nonsym_3x3".into(),
                a: vec![
                    vec![1.0, 2.0, 0.0],
                    vec![0.5, 1.0, 1.5],
                    vec![0.0, 0.25, 2.0],
                ],
                is_spd: false,
                is_square_invertible: true,
            },
            // 4×3 rectangular (only pinv applies)
            PointCase {
                case_id: "rect_4x3".into(),
                a: vec![
                    vec![1.0, 0.5, 0.0],
                    vec![0.0, 1.0, 0.25],
                    vec![0.5, 0.0, 1.0],
                    vec![1.0, 1.0, 0.5],
                ],
                is_spd: false,
                is_square_invertible: false,
            },
        ],
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.linalg import inv as scipy_inv, pinv as scipy_pinv, cholesky as scipy_chol

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

def listify_2d(arr):
    out = []
    for row in arr.tolist():
        rrow = []
        for v in row:
            f = fnone(v)
            if f is None:
                return None
            rrow.append(f)
        out.append(rrow)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    A = np.asarray(case["a"], dtype=np.float64)
    out = {"case_id": cid, "inv": None, "pinv": None, "chol_lower": None}
    try:
        if case["is_square_invertible"]:
            out["inv"] = listify_2d(np.asarray(scipy_inv(A), dtype=np.float64))
        out["pinv"] = listify_2d(np.asarray(scipy_pinv(A), dtype=np.float64))
        if case["is_spd"]:
            out["chol_lower"] = listify_2d(
                np.asarray(scipy_chol(A, lower=True), dtype=np.float64)
            )
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}, allow_nan=False))
"#;
    let query_json = serde_json::to_string(query).expect("serialize inv_pinv_cholesky query");
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
                "failed to spawn python3 for inv_pinv_cholesky oracle: {e}"
            );
            eprintln!("skipping inv_pinv_cholesky oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open inv_pinv_cholesky oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "inv_pinv_cholesky oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping inv_pinv_cholesky oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for inv_pinv_cholesky oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "inv_pinv_cholesky oracle failed: {stderr}"
        );
        eprintln!("skipping inv_pinv_cholesky oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse inv_pinv_cholesky oracle JSON"))
}

#[test]
fn diff_linalg_inv_pinv_cholesky() {
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
    // Each arm covers the cases its function applies to (the same flags that select what the
    // oracle computes).
    let mut ledger = CompareLedger::new(
        "diff_linalg_inv_pinv_cholesky",
        &["inv", "pinv", "cholesky_lower"],
    );

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");

        // inv
        if case.is_square_invertible {
            let rust_res = inv(&case.a, InvOptions::default()).ok();
            if let Some(max_d) = matrix_diff(
                &mut ledger,
                "inv",
                &case.case_id,
                scipy_arm.inv.as_deref(),
                rust_res.as_ref().map(|r| r.inverse.as_slice()),
            ) {
                max_overall = max_overall.max(max_d);
                ledger.compared("inv", &case.case_id, max_d <= ABS_TOL);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    sub_check: "inv".into(),
                    max_abs_diff: max_d,
                    pass: max_d <= ABS_TOL,
                });
            }
        }

        // pinv (every case)
        let rust_res = pinv(&case.a, PinvOptions::default()).ok();
        if let Some(max_d) = matrix_diff(
            &mut ledger,
            "pinv",
            &case.case_id,
            scipy_arm.pinv.as_deref(),
            rust_res.as_ref().map(|r| r.pseudo_inverse.as_slice()),
        ) {
            max_overall = max_overall.max(max_d);
            ledger.compared("pinv", &case.case_id, max_d <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "pinv".into(),
                max_abs_diff: max_d,
                pass: max_d <= ABS_TOL,
            });
        }

        // cholesky (lower)
        if case.is_spd {
            let rust_res = cholesky(&case.a, true, DecompOptions::default()).ok();
            if let Some(max_d) = matrix_diff(
                &mut ledger,
                "cholesky_lower",
                &case.case_id,
                scipy_arm.chol_lower.as_deref(),
                rust_res.as_ref().map(|r| r.factor.as_slice()),
            ) {
                max_overall = max_overall.max(max_d);
                ledger.compared("cholesky_lower", &case.case_id, max_d <= ABS_TOL);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    sub_check: "cholesky_lower".into(),
                    max_abs_diff: max_d,
                    pass: max_d <= ABS_TOL,
                });
            }
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_linalg_inv_pinv_cholesky".into(),
        category: "fsci_linalg::{inv,pinv,cholesky}".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
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
                "inv_pinv_cholesky mismatch: {} {} — max_abs={}",
                d.case_id, d.sub_check, d.max_abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "inv_pinv_cholesky conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
    // inv covers the square invertible cases, pinv every case, cholesky the SPD ones.
    let inv_cases = query
        .points
        .iter()
        .filter(|c| c.is_square_invertible)
        .count();
    let chol_cases = query.points.iter().filter(|c| c.is_spd).count();
    ledger.finish(inv_cases.min(chol_cases).min(query.points.len()));
}
