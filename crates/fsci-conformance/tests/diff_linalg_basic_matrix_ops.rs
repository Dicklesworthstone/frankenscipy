#![forbid(unsafe_code)]
//! Live numpy parity for fsci_linalg basic matrix/vector ops.
//!
//! Resolves [frankenscipy-1wlal]. Covers:
//!   * diag(M) extracts main diagonal as 1D vector
//!   * diagm(v) builds diagonal matrix from vector
//!   * eye(n, m) builds rectangular identity matrix
//!   * matmul(A, B) matrix-matrix multiplication
//!   * matvec(A, x) matrix-vector multiplication
//!   * outer(a, b) vector outer product
//!
//! Error paths: matmul/matvec with incompatible shapes.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{diag, diagm, eye, matmul, matvec, outer};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    /// "diag" | "diagm" | "eye" | "matmul" | "matvec" | "outer"
    op: String,
    a: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    vec_a: Vec<f64>,
    vec_b: Vec<f64>,
    n: usize,
    m: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<CasePoint>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OraclePoint {
    case_id: String,
    rows: Option<usize>,
    cols: Option<usize>,
    data: Option<Vec<f64>>,
    /// For diag (1D output)
    flat: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OraclePoint>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    max_abs_diff: f64,
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
    fs::create_dir_all(output_dir()).expect("create matops diff dir");
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
    let m_3x3 = vec![
        vec![1.0_f64, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let m_2x3 = vec![vec![1.0_f64, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let m_3x2 = vec![vec![1.0_f64, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let v_3 = vec![1.0_f64, 2.0, 3.0];
    let v_4 = vec![0.5_f64, 1.5, 2.5, 3.5];

    let pts = vec![
        CasePoint {
            case_id: "diag_3x3".into(),
            op: "diag".into(),
            a: m_3x3.clone(),
            b: Vec::new(),
            vec_a: Vec::new(),
            vec_b: Vec::new(),
            n: 0,
            m: 0,
        },
        CasePoint {
            case_id: "diag_2x3_rect".into(),
            op: "diag".into(),
            a: m_2x3.clone(),
            b: Vec::new(),
            vec_a: Vec::new(),
            vec_b: Vec::new(),
            n: 0,
            m: 0,
        },
        CasePoint {
            case_id: "diagm_3".into(),
            op: "diagm".into(),
            a: Vec::new(),
            b: Vec::new(),
            vec_a: v_3.clone(),
            vec_b: Vec::new(),
            n: 0,
            m: 0,
        },
        CasePoint {
            case_id: "eye_5x5".into(),
            op: "eye".into(),
            a: Vec::new(),
            b: Vec::new(),
            vec_a: Vec::new(),
            vec_b: Vec::new(),
            n: 5,
            m: 5,
        },
        CasePoint {
            case_id: "eye_3x5".into(),
            op: "eye".into(),
            a: Vec::new(),
            b: Vec::new(),
            vec_a: Vec::new(),
            vec_b: Vec::new(),
            n: 3,
            m: 5,
        },
        CasePoint {
            case_id: "eye_5x3".into(),
            op: "eye".into(),
            a: Vec::new(),
            b: Vec::new(),
            vec_a: Vec::new(),
            vec_b: Vec::new(),
            n: 5,
            m: 3,
        },
        CasePoint {
            case_id: "matmul_3x3_3x3".into(),
            op: "matmul".into(),
            a: m_3x3.clone(),
            b: m_3x3.clone(),
            vec_a: Vec::new(),
            vec_b: Vec::new(),
            n: 0,
            m: 0,
        },
        CasePoint {
            case_id: "matmul_2x3_3x2".into(),
            op: "matmul".into(),
            a: m_2x3.clone(),
            b: m_3x2.clone(),
            vec_a: Vec::new(),
            vec_b: Vec::new(),
            n: 0,
            m: 0,
        },
        CasePoint {
            case_id: "matvec_3x3_v3".into(),
            op: "matvec".into(),
            a: m_3x3.clone(),
            b: Vec::new(),
            vec_a: v_3.clone(),
            vec_b: Vec::new(),
            n: 0,
            m: 0,
        },
        CasePoint {
            case_id: "matvec_2x3_v3".into(),
            op: "matvec".into(),
            a: m_2x3,
            b: Vec::new(),
            vec_a: v_3.clone(),
            vec_b: Vec::new(),
            n: 0,
            m: 0,
        },
        CasePoint {
            case_id: "outer_v3_v4".into(),
            op: "outer".into(),
            a: Vec::new(),
            b: Vec::new(),
            vec_a: v_3,
            vec_b: v_4.clone(),
            n: 0,
            m: 0,
        },
    ];

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
        if op == "diag":
            m = np.array(c["a"], dtype=float)
            v = np.diag(m)
            out.append({"case_id": cid, "rows": None, "cols": None, "data": None,
                        "flat": [float(x) for x in v]})
        elif op == "diagm":
            v = np.array(c["vec_a"], dtype=float)
            m = np.diag(v)
            out.append({"case_id": cid, "rows": int(m.shape[0]), "cols": int(m.shape[1]),
                        "data": [float(x) for x in m.flatten()], "flat": None})
        elif op == "eye":
            m = np.eye(int(c["n"]), int(c["m"]))
            out.append({"case_id": cid, "rows": int(m.shape[0]), "cols": int(m.shape[1]),
                        "data": [float(x) for x in m.flatten()], "flat": None})
        elif op == "matmul":
            a = np.array(c["a"], dtype=float)
            b = np.array(c["b"], dtype=float)
            m = a @ b
            out.append({"case_id": cid, "rows": int(m.shape[0]), "cols": int(m.shape[1]),
                        "data": [float(x) for x in m.flatten()], "flat": None})
        elif op == "matvec":
            a = np.array(c["a"], dtype=float)
            x = np.array(c["vec_a"], dtype=float)
            v = a @ x
            out.append({"case_id": cid, "rows": None, "cols": None, "data": None,
                        "flat": [float(x) for x in v]})
        elif op == "outer":
            a = np.array(c["vec_a"], dtype=float)
            b = np.array(c["vec_b"], dtype=float)
            m = np.outer(a, b)
            out.append({"case_id": cid, "rows": int(m.shape[0]), "cols": int(m.shape[1]),
                        "data": [float(x) for x in m.flatten()], "flat": None})
        else:
            out.append({"case_id": cid, "rows": None, "cols": None, "data": None, "flat": None})
    except Exception:
        out.append({"case_id": cid, "rows": None, "cols": None, "data": None, "flat": None})

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
            eprintln!("skipping matops oracle: python3 unavailable ({e})");
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
            eprintln!("skipping matops oracle: stdin write failed");
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
        eprintln!("skipping matops oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

fn flatten(m: &[Vec<f64>]) -> (usize, usize, Vec<f64>) {
    let rows = m.len();
    let cols = m.first().map_or(0, |r| r.len());
    let mut flat = Vec::with_capacity(rows * cols);
    for row in m {
        flat.extend_from_slice(row);
    }
    (rows, cols, flat)
}

#[test]
fn diff_linalg_basic_matrix_ops() {
    let query = build_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    for (case, o) in query.points.iter().zip(oracle.points.iter()) {
        assert_eq!(case.case_id, o.case_id);
        match case.op.as_str() {
            "diag" => {
                let Some(exp_flat) = o.flat.as_ref() else {
                    continue;
                };
                let actual = diag(&case.a);
                if actual.len() != exp_flat.len() {
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        op: case.op.clone(),
                        max_abs_diff: f64::INFINITY,
                        pass: false,
                        note: format!("len mismatch: {} vs {}", actual.len(), exp_flat.len()),
                    });
                    continue;
                }
                let max_abs = actual
                    .iter()
                    .zip(exp_flat.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    max_abs_diff: max_abs,
                    pass: max_abs <= ABS_TOL,
                    note: String::new(),
                });
            }
            "matvec" => {
                let Some(exp_flat) = o.flat.as_ref() else {
                    continue;
                };
                let actual = match matvec(&case.a, &case.vec_a) {
                    Ok(v) => v,
                    Err(e) => {
                        diffs.push(CaseDiff {
                            case_id: case.case_id.clone(),
                            op: case.op.clone(),
                            max_abs_diff: f64::INFINITY,
                            pass: false,
                            note: format!("error: {e:?}"),
                        });
                        continue;
                    }
                };
                let max_abs = actual
                    .iter()
                    .zip(exp_flat.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    max_abs_diff: max_abs,
                    pass: max_abs <= ABS_TOL,
                    note: String::new(),
                });
            }
            op => {
                let (Some(exp_rows), Some(exp_cols), Some(exp_data)) =
                    (o.rows, o.cols, o.data.as_ref())
                else {
                    continue;
                };
                let actual_mat: Vec<Vec<f64>> = match op {
                    "diagm" => diagm(&case.vec_a),
                    "eye" => eye(case.n, case.m),
                    "matmul" => match matmul(&case.a, &case.b) {
                        Ok(v) => v,
                        Err(e) => {
                            diffs.push(CaseDiff {
                                case_id: case.case_id.clone(),
                                op: case.op.clone(),
                                max_abs_diff: f64::INFINITY,
                                pass: false,
                                note: format!("matmul error: {e:?}"),
                            });
                            continue;
                        }
                    },
                    "outer" => outer(&case.vec_a, &case.vec_b),
                    other => panic!("unknown op {other}"),
                };
                let (rows, cols, flat) = flatten(&actual_mat);
                if rows != exp_rows || cols != exp_cols {
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        op: case.op.clone(),
                        max_abs_diff: f64::INFINITY,
                        pass: false,
                        note: format!("shape mismatch: {rows}x{cols} vs {exp_rows}x{exp_cols}"),
                    });
                    continue;
                }
                let max_abs = flat
                    .iter()
                    .zip(exp_data.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    max_abs_diff: max_abs,
                    pass: max_abs <= ABS_TOL,
                    note: String::new(),
                });
            }
        }
    }

    // === Error paths ===
    // matmul incompatible shapes
    {
        let a = vec![vec![1.0_f64, 2.0]; 3]; // 3x2
        let b = vec![vec![1.0_f64; 4]; 5]; // 5x4 (rows must equal a's cols=2)
        let r = matmul(&a, &b);
        diffs.push(CaseDiff {
            case_id: "matmul_incompatible_shapes_errors".into(),
            op: "matmul".into(),
            max_abs_diff: 0.0,
            pass: r.is_err(),
            note: format!("res={r:?}"),
        });
    }
    // matvec length mismatch
    {
        let a = vec![vec![1.0_f64, 2.0]; 3]; // 3x2
        let x = vec![1.0_f64, 2.0, 3.0]; // length 3 ≠ 2
        let r = matvec(&a, &x);
        diffs.push(CaseDiff {
            case_id: "matvec_length_mismatch_errors".into(),
            op: "matvec".into(),
            max_abs_diff: 0.0,
            pass: r.is_err(),
            note: format!("res={r:?}"),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_linalg_basic_matrix_ops".into(),
        category: "fsci_linalg::{diag, diagm, eye, matmul, matvec, outer} numpy parity".into(),
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
                "matops mismatch: {} ({}) max_abs={} note={}",
                d.case_id, d.op, d.max_abs_diff, d.note
            );
        }
    }

    assert!(all_pass, "matrix-ops parity failed: {} cases", diffs.len());
}
