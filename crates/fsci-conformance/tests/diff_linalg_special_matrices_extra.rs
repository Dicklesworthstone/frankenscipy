#![forbid(unsafe_code)]
//! Live scipy parity for fsci_linalg structured matrix constructors.
//!
//! Resolves [frankenscipy-x2oox]. Covers a batch of small-but-uncovered
//! structured-matrix builders against scipy.linalg equivalents:
//!   * leslie(f, s)   vs scipy.linalg.leslie
//!   * pascal(n, sym) vs scipy.linalg.pascal(n, kind='symmetric'|'lower')
//!   * vander(x, n, increasing) vs numpy.vander
//!   * hankel(c, r)   vs scipy.linalg.hankel (with r=None → zeros default)
//!   * helmert(n)     vs scipy.linalg.helmert(n, full=False)
//!   * helmert_full(n) vs scipy.linalg.helmert(n, full=True)

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{hankel, helmert, helmert_full, leslie, pascal, vander};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-10;
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    /// "leslie" | "pascal_sym" | "pascal_lower" | "vander_inc" | "vander_dec"
    /// | "hankel_with_r" | "hankel_zero_r" | "helmert_sub" | "helmert_full"
    op: String,
    /// First args (vec input)
    a: Vec<f64>,
    /// Second args (vec input for r in hankel, s in leslie)
    b: Vec<f64>,
    /// Integer arg (n, vander columns, etc.)
    n: usize,
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
    /// Flattened row-major
    data: Option<Vec<f64>>,
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
    max_rel_diff: f64,
    rows: usize,
    cols: usize,
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
    fs::create_dir_all(output_dir()).expect("create special_matrices diff dir");
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

    // leslie: f and s vectors with len(s) == len(f) - 1
    pts.push(CasePoint {
        case_id: "leslie_4".into(),
        op: "leslie".into(),
        a: vec![0.1, 2.0, 1.0, 0.1],
        b: vec![0.2, 0.8, 0.7],
        n: 0,
    });
    pts.push(CasePoint {
        case_id: "leslie_3".into(),
        op: "leslie".into(),
        a: vec![1.5, 0.8, 0.2],
        b: vec![0.6, 0.4],
        n: 0,
    });

    // pascal symmetric
    for &n in &[1_usize, 3, 5, 7] {
        pts.push(CasePoint {
            case_id: format!("pascal_sym_n{n}"),
            op: "pascal_sym".into(),
            a: Vec::new(),
            b: Vec::new(),
            n,
        });
        pts.push(CasePoint {
            case_id: format!("pascal_lower_n{n}"),
            op: "pascal_lower".into(),
            a: Vec::new(),
            b: Vec::new(),
            n,
        });
    }

    // vander (increasing and decreasing)
    pts.push(CasePoint {
        case_id: "vander_default_dec_n5".into(),
        op: "vander_dec".into(),
        a: vec![1.0, 2.0, 3.0, 4.0],
        b: Vec::new(),
        n: 5,
    });
    pts.push(CasePoint {
        case_id: "vander_inc_n5".into(),
        op: "vander_inc".into(),
        a: vec![1.0, 2.0, 3.0, 4.0],
        b: Vec::new(),
        n: 5,
    });
    pts.push(CasePoint {
        case_id: "vander_default_dec_n3".into(),
        op: "vander_dec".into(),
        a: vec![0.5, 1.5, 2.5],
        b: Vec::new(),
        n: 3,
    });

    // hankel (with and without explicit r)
    pts.push(CasePoint {
        case_id: "hankel_c4_with_r".into(),
        op: "hankel_with_r".into(),
        a: vec![1.0, 2.0, 3.0, 4.0],
        b: vec![4.0, 7.0, 7.0, 8.0],
        n: 0,
    });
    pts.push(CasePoint {
        case_id: "hankel_c4_zero_r".into(),
        op: "hankel_zero_r".into(),
        a: vec![1.0, 2.0, 3.0, 4.0],
        b: Vec::new(),
        n: 0,
    });

    // helmert
    for &n in &[2_usize, 3, 4, 5, 8] {
        pts.push(CasePoint {
            case_id: format!("helmert_sub_n{n}"),
            op: "helmert_sub".into(),
            a: Vec::new(),
            b: Vec::new(),
            n,
        });
        pts.push(CasePoint {
            case_id: format!("helmert_full_n{n}"),
            op: "helmert_full".into(),
            a: Vec::new(),
            b: Vec::new(),
            n,
        });
    }

    OracleQuery { points: pts }
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
import numpy as np
from scipy import linalg

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    cid = c["case_id"]
    op = c["op"]
    a = np.array(c["a"], dtype=float)
    b = np.array(c["b"], dtype=float)
    n = int(c["n"])
    try:
        if op == "leslie":
            m = linalg.leslie(a, b)
        elif op == "pascal_sym":
            m = linalg.pascal(n, kind='symmetric')
        elif op == "pascal_lower":
            m = linalg.pascal(n, kind='lower')
        elif op == "vander_inc":
            m = np.vander(a, N=n, increasing=True)
        elif op == "vander_dec":
            m = np.vander(a, N=n, increasing=False)
        elif op == "hankel_with_r":
            m = linalg.hankel(a, b)
        elif op == "hankel_zero_r":
            m = linalg.hankel(a, np.zeros_like(a))
        elif op == "helmert_sub":
            m = linalg.helmert(n, full=False)
        elif op == "helmert_full":
            m = linalg.helmert(n, full=True)
        else:
            m = None
        if m is None or not np.all(np.isfinite(m)):
            out.append({"case_id": cid, "rows": None, "cols": None, "data": None})
        else:
            m = np.asarray(m, dtype=float)
            rows, cols = m.shape
            out.append({
                "case_id": cid,
                "rows": int(rows),
                "cols": int(cols),
                "data": [float(v) for v in m.flatten()],
            })
    except Exception:
        out.append({"case_id": cid, "rows": None, "cols": None, "data": None})

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
            eprintln!("skipping special_matrices oracle: python3 unavailable ({e})");
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
            eprintln!("skipping special_matrices oracle: stdin write failed");
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
        eprintln!("skipping special_matrices oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

fn fsci_compute(case: &CasePoint) -> Result<Vec<Vec<f64>>, String> {
    match case.op.as_str() {
        "leslie" => leslie(&case.a, &case.b).map_err(|e| format!("{e:?}")),
        "pascal_sym" => Ok(pascal(case.n, true)),
        "pascal_lower" => Ok(pascal(case.n, false)),
        "vander_inc" => Ok(vander(&case.a, Some(case.n), true)),
        "vander_dec" => Ok(vander(&case.a, Some(case.n), false)),
        "hankel_with_r" => Ok(hankel(&case.a, Some(&case.b))),
        "hankel_zero_r" => Ok(hankel(&case.a, None)),
        "helmert_sub" => Ok(helmert(case.n)),
        "helmert_full" => Ok(helmert_full(case.n)),
        other => Err(format!("unknown op {other}")),
    }
}

#[test]
fn diff_linalg_special_matrices_extra() {
    let query = build_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    for (case, o) in query.points.iter().zip(oracle.points.iter()) {
        assert_eq!(case.case_id, o.case_id);
        let (Some(exp_rows), Some(exp_cols), Some(exp_data)) = (o.rows, o.cols, o.data.as_ref())
        else {
            continue;
        };

        let actual = match fsci_compute(case) {
            Ok(m) => m,
            Err(e) => {
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    max_abs_diff: f64::INFINITY,
                    max_rel_diff: f64::INFINITY,
                    rows: 0,
                    cols: 0,
                    pass: false,
                    note: e,
                });
                continue;
            }
        };
        let rows = actual.len();
        let cols = actual.first().map_or(0, |r| r.len());
        if rows != exp_rows || cols != exp_cols {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                op: case.op.clone(),
                max_abs_diff: f64::INFINITY,
                max_rel_diff: f64::INFINITY,
                rows,
                cols,
                pass: false,
                note: format!("shape mismatch: fsci {rows}x{cols} scipy {exp_rows}x{exp_cols}"),
            });
            continue;
        }

        let mut max_abs = 0.0_f64;
        let mut max_rel = 0.0_f64;
        for (i, row) in actual.iter().enumerate() {
            for (j, &a) in row.iter().enumerate() {
                let e = exp_data[i * cols + j];
                let abs_d = (a - e).abs();
                let denom = e.abs().max(1.0e-300);
                max_abs = max_abs.max(abs_d);
                max_rel = max_rel.max(abs_d / denom);
            }
        }
        let pass = max_rel <= REL_TOL || max_abs <= ABS_TOL;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            max_abs_diff: max_abs,
            max_rel_diff: max_rel,
            rows,
            cols,
            pass,
            note: String::new(),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_linalg_special_matrices_extra".into(),
        category: "fsci_linalg::{leslie, pascal, vander, hankel, helmert*} vs scipy.linalg/numpy"
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
                "special_matrices mismatch: {} ({}) max_rel={} max_abs={} shape={}x{} note={}",
                d.case_id, d.op, d.max_rel_diff, d.max_abs_diff, d.rows, d.cols, d.note
            );
        }
    }

    assert!(
        all_pass,
        "special-matrices parity failed: {} cases",
        diffs.len()
    );
}
