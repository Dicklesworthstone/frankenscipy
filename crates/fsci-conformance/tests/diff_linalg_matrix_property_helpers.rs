#![forbid(unsafe_code)]
//! Live numpy/scipy parity for fsci_linalg matrix property helpers.
//!
//! Resolves [frankenscipy-3m6s4]. Covers scalar/predicate helpers
//! that previously had no dedicated diff:
//!   * mat_norm_1 (max column sum) vs np.linalg.norm(..., ord=1)
//!   * mat_norm_inf (max row sum) vs np.linalg.norm(..., ord=np.inf)
//!   * trace (sum of diagonal) vs np.trace
//!   * cond (condition number) vs np.linalg.cond
//!   * is_diagonal / is_upper_triangular / is_lower_triangular /
//!     is_orthogonal: boolean predicates verified on hand-built
//!     matrices that exercise both true and false branches

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{
    DecompOptions, cond, is_diagonal, is_lower_triangular, is_orthogonal, is_upper_triangular,
    mat_norm_1, mat_norm_inf, trace,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-10;
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    /// "norm_1" | "norm_inf" | "trace" | "cond"
    op: String,
    /// Square or rectangular matrix as Vec<Vec<f64>>
    matrix: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<CasePoint>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OraclePoint {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OraclePoint>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    actual: f64,
    expected: f64,
    abs_diff: f64,
    rel_diff: f64,
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
    fs::create_dir_all(output_dir()).expect("create matprop diff dir");
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
    let m_3x3_general = vec![
        vec![1.0_f64, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.5],
    ];
    let m_4x3_rect = vec![
        vec![1.0_f64, -2.0, 3.0],
        vec![-4.0, 5.0, -6.0],
        vec![7.0, -8.0, 9.0],
        vec![-1.0, 2.0, -3.0],
    ];
    let m_2x2_neg = vec![vec![-1.0_f64, -2.0], vec![3.0, 4.0]];
    let m_5x5_diag: Vec<Vec<f64>> = (0..5)
        .map(|i| {
            (0..5)
                .map(|j| if i == j { (i + 1) as f64 } else { 0.0 })
                .collect()
        })
        .collect();

    let mut pts = Vec::new();
    for (label, m) in [
        ("general_3x3", m_3x3_general.clone()),
        ("rect_4x3", m_4x3_rect.clone()),
        ("with_negatives_2x2", m_2x2_neg.clone()),
        ("diag_5x5", m_5x5_diag.clone()),
    ] {
        for op in ["norm_1", "norm_inf"] {
            pts.push(CasePoint {
                case_id: format!("{op}_{label}"),
                op: op.into(),
                matrix: m.clone(),
            });
        }
    }
    // trace on square matrices
    for (label, m) in [
        ("general_3x3", m_3x3_general.clone()),
        ("with_negatives_2x2", m_2x2_neg.clone()),
        ("diag_5x5", m_5x5_diag.clone()),
    ] {
        pts.push(CasePoint {
            case_id: format!("trace_{label}"),
            op: "trace".into(),
            matrix: m,
        });
    }
    // cond on well-conditioned square matrices (skip near-singular to avoid
    // numerical drift between fsci and numpy's SVD).
    pts.push(CasePoint {
        case_id: "cond_diag_5x5".into(),
        op: "cond".into(),
        matrix: m_5x5_diag.clone(),
    });
    pts.push(CasePoint {
        case_id: "cond_general_3x3".into(),
        op: "cond".into(),
        matrix: m_3x3_general,
    });

    OracleQuery { points: pts }
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
import numpy as np

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    cid = c["case_id"]
    op = c["op"]
    try:
        m = np.array(c["matrix"], dtype=float)
        if op == "norm_1":
            v = float(np.linalg.norm(m, ord=1))
        elif op == "norm_inf":
            v = float(np.linalg.norm(m, ord=np.inf))
        elif op == "trace":
            v = float(np.trace(m))
        elif op == "cond":
            v = float(np.linalg.cond(m))
        else:
            v = None
        if v is None or not math.isfinite(v):
            out.append({"case_id": cid, "value": None})
        else:
            out.append({"case_id": cid, "value": v})
    except Exception:
        out.append({"case_id": cid, "value": None})

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
            eprintln!("skipping matprop oracle: python3 unavailable ({e})");
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
            eprintln!("skipping matprop oracle: stdin write failed");
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
        eprintln!("skipping matprop oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

fn fsci_compute(case: &CasePoint) -> Result<f64, String> {
    match case.op.as_str() {
        "norm_1" => Ok(mat_norm_1(&case.matrix)),
        "norm_inf" => Ok(mat_norm_inf(&case.matrix)),
        "trace" => Ok(trace(&case.matrix)),
        "cond" => cond(&case.matrix, DecompOptions::default()).map_err(|e| format!("{e:?}")),
        other => Err(format!("unknown op {other}")),
    }
}

#[test]
fn diff_linalg_matrix_property_helpers() {
    let query = build_query();
    let oracle_opt = scipy_oracle_or_skip(&query);

    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    // === Numeric helpers: compared against numpy ===
    if let Some(oracle) = oracle_opt {
        assert_eq!(oracle.points.len(), query.points.len());
        for (case, o) in query.points.iter().zip(oracle.points.iter()) {
            assert_eq!(case.case_id, o.case_id);
            let Some(expected) = o.value else {
                continue;
            };
            let actual = match fsci_compute(case) {
                Ok(v) => v,
                Err(e) => {
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        op: case.op.clone(),
                        actual: f64::NAN,
                        expected,
                        abs_diff: f64::INFINITY,
                        rel_diff: f64::INFINITY,
                        pass: false,
                        note: e,
                    });
                    continue;
                }
            };
            let abs_diff = (actual - expected).abs();
            let denom = expected.abs().max(1e-300);
            let rel_diff = abs_diff / denom;
            let pass = rel_diff <= REL_TOL || abs_diff <= ABS_TOL;
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                op: case.op.clone(),
                actual,
                expected,
                abs_diff,
                rel_diff,
                pass,
                note: String::new(),
            });
        }
    }

    // === Predicate helpers: in-Rust expected values ===
    let mut pred_check = |id: &str, ok: bool, note: String| {
        diffs.push(CaseDiff {
            case_id: id.into(),
            op: "predicate".into(),
            actual: f64::NAN,
            expected: f64::NAN,
            abs_diff: 0.0,
            rel_diff: 0.0,
            pass: ok,
            note,
        });
    };

    let diag_3 = vec![
        vec![1.0_f64, 0.0, 0.0],
        vec![0.0, 2.0, 0.0],
        vec![0.0, 0.0, 3.0],
    ];
    let upper_3 = vec![
        vec![1.0_f64, 2.0, 3.0],
        vec![0.0, 4.0, 5.0],
        vec![0.0, 0.0, 6.0],
    ];
    let lower_3 = vec![
        vec![1.0_f64, 0.0, 0.0],
        vec![2.0, 3.0, 0.0],
        vec![4.0, 5.0, 6.0],
    ];
    let full_3 = vec![
        vec![1.0_f64, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    // 2D rotation matrix (orthogonal)
    let theta = std::f64::consts::PI / 6.0;
    let rot_2 = vec![
        vec![theta.cos(), -theta.sin()],
        vec![theta.sin(), theta.cos()],
    ];

    let tol = 1.0e-10_f64;
    pred_check(
        "diag_3_is_diagonal_true",
        is_diagonal(&diag_3, tol),
        String::new(),
    );
    pred_check(
        "upper_3_is_diagonal_false",
        !is_diagonal(&upper_3, tol),
        String::new(),
    );
    pred_check(
        "upper_3_is_upper_triangular_true",
        is_upper_triangular(&upper_3, tol),
        String::new(),
    );
    pred_check(
        "lower_3_is_upper_triangular_false",
        !is_upper_triangular(&lower_3, tol),
        String::new(),
    );
    pred_check(
        "lower_3_is_lower_triangular_true",
        is_lower_triangular(&lower_3, tol),
        String::new(),
    );
    pred_check(
        "upper_3_is_lower_triangular_false",
        !is_lower_triangular(&upper_3, tol),
        String::new(),
    );
    pred_check(
        "diag_3_is_both_triangular",
        is_upper_triangular(&diag_3, tol) && is_lower_triangular(&diag_3, tol),
        String::new(),
    );
    pred_check(
        "rot_2_is_orthogonal_true",
        is_orthogonal(&rot_2, tol),
        String::new(),
    );
    pred_check(
        "full_3_is_orthogonal_false",
        !is_orthogonal(&full_3, tol),
        String::new(),
    );

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_linalg_matrix_property_helpers".into(),
        category:
            "fsci_linalg::{mat_norm_1, mat_norm_inf, trace, cond, is_diagonal, is_upper_triangular, is_lower_triangular, is_orthogonal} coverage"
                .into(),
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
                "matprop mismatch: {} ({}) actual={} expected={} abs={} rel={} note={}",
                d.case_id, d.op, d.actual, d.expected, d.abs_diff, d.rel_diff, d.note
            );
        }
    }

    assert!(
        all_pass,
        "matrix property helper coverage failed: {} cases",
        diffs.len()
    );
}
