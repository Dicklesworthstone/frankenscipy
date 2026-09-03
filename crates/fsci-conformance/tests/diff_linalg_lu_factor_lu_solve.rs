#![forbid(unsafe_code)]
//! Live scipy parity for fsci_linalg::{lu_factor, lu_solve}.
//!
//! Resolves [frankenscipy-c9k7y]. Verifies the factor-then-solve pair
//! matches scipy.linalg.lu_solve(scipy.linalg.lu_factor(A), b) for
//! several well-conditioned, ill-conditioned, banded, and unsymmetric
//! systems. Also verifies the solution residual ||A x − b||∞ stays
//! within a tight bound (LU is a direct solver, so residuals should
//! be at the machine-precision level for well-conditioned matrices).

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{DecompOptions, lu_factor, lu_solve};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-8;
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    /// Square matrix stored row-major as Vec<Vec<f64>>
    a: Vec<Vec<f64>>,
    /// Right-hand side
    b: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<CasePoint>,
}

#[derive(Debug, Clone, Deserialize)]
struct OraclePoint {
    case_id: String,
    /// scipy.linalg.lu_solve(lu_factor(a), b) — solution vector
    x: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OraclePoint>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    max_abs_diff: f64,
    max_rel_diff: f64,
    residual_inf: f64,
    pass: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create lu_factor diff dir");
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

fn build_query() -> OracleQuery {
    let mut pts = Vec::new();

    // 1. 3x3 diagonally-dominant well-conditioned
    pts.push(CasePoint {
        case_id: "ddom_3x3".into(),
        a: vec![
            vec![4.0_f64, -1.0, 0.0],
            vec![-1.0, 4.0, -1.0],
            vec![0.0, -1.0, 3.0],
        ],
        b: vec![1.0, 2.0, 3.0],
    });

    // 2. 4x4 unsymmetric general
    pts.push(CasePoint {
        case_id: "general_4x4".into(),
        a: vec![
            vec![2.0_f64, 1.0, 0.0, 0.0],
            vec![1.0, 3.0, 1.0, 0.0],
            vec![0.0, 1.0, 4.0, 1.0],
            vec![0.0, 0.0, 1.0, 2.0],
        ],
        b: vec![1.0, 4.0, 6.0, 3.0],
    });

    // 3. 5x5 tridiagonal banded
    let mut a = vec![vec![0.0_f64; 5]; 5];
    for i in 0..5 {
        a[i][i] = 4.0;
        if i + 1 < 5 {
            a[i][i + 1] = -1.0;
            a[i + 1][i] = -1.0;
        }
    }
    pts.push(CasePoint {
        case_id: "tridiag_5x5".into(),
        a,
        b: vec![1.0, 1.0, 1.0, 1.0, 1.0],
    });

    // 4. 3x3 unsymmetric with large off-diagonal
    pts.push(CasePoint {
        case_id: "unsym_3x3".into(),
        a: vec![
            vec![1.0_f64, 7.0, -3.0],
            vec![2.0, 1.0, 4.0],
            vec![-3.0, 2.0, 1.0],
        ],
        b: vec![5.0, 7.0, 0.0],
    });

    // 5. Hilbert-like 4x4 (ill-conditioned but invertible)
    let h: Vec<Vec<f64>> = (1..=4)
        .map(|i| (1..=4).map(|j| 1.0 / (i as f64 + j as f64 - 1.0)).collect())
        .collect();
    pts.push(CasePoint {
        case_id: "hilbert_4x4".into(),
        a: h,
        b: vec![1.0, 1.0, 1.0, 1.0],
    });

    // 6. Random-ish 6x6 with mixed signs (deterministic seed values)
    pts.push(CasePoint {
        case_id: "mixed_6x6".into(),
        a: vec![
            vec![3.0_f64, -2.0, 1.0, 0.0, 0.5, -1.0],
            vec![1.0, 4.0, -2.0, 1.0, 0.0, 0.5],
            vec![-1.0, 1.0, 5.0, -2.0, 1.0, 0.0],
            vec![0.0, -1.0, 1.0, 4.0, -2.0, 1.0],
            vec![0.5, 0.0, -1.0, 1.0, 3.0, -2.0],
            vec![-1.0, 0.5, 0.0, -1.0, 1.0, 4.0],
        ],
        b: vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0],
    });

    // 7. 2x2 small
    pts.push(CasePoint {
        case_id: "small_2x2".into(),
        a: vec![vec![3.0_f64, 2.0], vec![1.0, 4.0]],
        b: vec![7.0, 5.0],
    });

    // 8. 1x1 trivial
    pts.push(CasePoint {
        case_id: "scalar_1x1".into(),
        a: vec![vec![5.0_f64]],
        b: vec![10.0],
    });

    OracleQuery { points: pts }
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
import numpy as np
from scipy.linalg import lu_factor, lu_solve

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    cid = c["case_id"]
    try:
        a = np.array(c["a"], dtype=float)
        b = np.array(c["b"], dtype=float)
        lup = lu_factor(a)
        x = lu_solve(lup, b)
        if not np.all(np.isfinite(x)):
            out.append({"case_id": cid, "x": None})
        else:
            out.append({"case_id": cid, "x": [float(v) for v in x]})
    except Exception as e:
        out.append({"case_id": cid, "x": None})

print(json.dumps({"points": out}))
"#;
    let query_json = serde_json::to_string(q).expect("serialize");
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
                "python3 spawn failed: {e}"
            );
            eprintln!("skipping lu_factor oracle: python3 unavailable ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping lu_factor oracle: stdin write failed");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "oracle failed: {stderr}"
        );
        eprintln!("skipping lu_factor oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

#[test]
fn diff_linalg_lu_factor_lu_solve() {
    let query = build_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    for (case, o) in query.points.iter().zip(oracle.points.iter()) {
        assert_eq!(case.case_id, o.case_id);
        let Some(expected) = o.x.as_ref() else {
            continue;
        };

        let opts = DecompOptions::default();
        let factored = match lu_factor(&case.a, opts) {
            Ok(f) => f,
            Err(e) => {
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    max_abs_diff: f64::INFINITY,
                    max_rel_diff: f64::INFINITY,
                    residual_inf: f64::INFINITY,
                    pass: false,
                    note: format!("lu_factor error: {e:?}"),
                });
                continue;
            }
        };
        let sol = match lu_solve(&factored, &case.b) {
            Ok(s) => s.x,
            Err(e) => {
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    max_abs_diff: f64::INFINITY,
                    max_rel_diff: f64::INFINITY,
                    residual_inf: f64::INFINITY,
                    pass: false,
                    note: format!("lu_solve error: {e:?}"),
                });
                continue;
            }
        };

        if sol.len() != expected.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                max_abs_diff: f64::INFINITY,
                max_rel_diff: f64::INFINITY,
                residual_inf: f64::INFINITY,
                pass: false,
                note: format!(
                    "length mismatch: fsci={} scipy={}",
                    sol.len(),
                    expected.len()
                ),
            });
            continue;
        }

        let mut max_abs = 0.0_f64;
        let mut max_rel = 0.0_f64;
        for (a, e) in sol.iter().zip(expected.iter()) {
            let abs_d = (a - e).abs();
            let denom = e.abs().max(1.0e-300);
            max_abs = max_abs.max(abs_d);
            max_rel = max_rel.max(abs_d / denom);
        }
        // Residual: A x - b
        let ax = matvec(&case.a, &sol);
        let mut residual_inf = 0.0_f64;
        for (ax_i, b_i) in ax.iter().zip(case.b.iter()) {
            residual_inf = residual_inf.max((ax_i - b_i).abs());
        }

        // Pass: (1) close to scipy solution and (2) residual is tight
        let close_to_scipy = max_rel <= REL_TOL || max_abs <= ABS_TOL;
        // Hilbert is famously ill-conditioned; relax residual tolerance for it.
        let res_tol = if case.case_id.starts_with("hilbert_") {
            1.0e-6
        } else {
            1.0e-10
        };
        let small_residual = residual_inf <= res_tol;
        let pass = close_to_scipy && small_residual;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            max_abs_diff: max_abs,
            max_rel_diff: max_rel,
            residual_inf,
            pass,
            note: String::new(),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_linalg_lu_factor_lu_solve".into(),
        category: "fsci_linalg::{lu_factor, lu_solve} vs scipy.linalg".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "lu_factor mismatch: {} max_rel={} max_abs={} res={} note={}",
                d.case_id, d.max_rel_diff, d.max_abs_diff, d.residual_inf, d.note
            );
        }
    }

    assert!(
        all_pass,
        "lu_factor/lu_solve parity failed: {} cases",
        diffs.len()
    );
}
