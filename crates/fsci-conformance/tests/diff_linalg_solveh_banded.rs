#![forbid(unsafe_code)]
//! Live scipy parity for fsci_linalg::solveh_banded.
//!
//! Resolves [frankenscipy-xm4fx]. Builds symmetric positive-definite
//! banded matrices (diagonal, tridiagonal, pentadiagonal) in both
//! lower and upper LAPACK band-storage formats, compares fsci's
//! solveh_banded against scipy.linalg.solveh_banded, and verifies the
//! solution residual ||A x − b||∞ stays tight when checked against
//! the dense reconstruction.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::solveh_banded;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-8;
const ABS_TOL: f64 = 1.0e-10;
const RESIDUAL_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    /// Banded storage matrix, rows = bandwidth+1, cols = n
    ab: Vec<Vec<f64>>,
    /// Right-hand side
    b: Vec<f64>,
    /// true => lower band storage; false => upper band storage
    lower: bool,
    /// Dense reconstruction for residual check (the test rebuilds A)
    /// — convenient to pass along rather than rebuild on Rust side
    a_dense: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<CasePoint>,
}

#[derive(Debug, Clone, Deserialize)]
struct OraclePoint {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create solveh_banded diff dir");
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

/// LAPACK lower-band storage of an SPD matrix A.
/// ab[k][j] = A[j+k][j] for k=0..bw, j=0..n-1-k. Unused slots set to 0.
fn lower_band_from_dense(a: &[Vec<f64>], bw: usize) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut ab = vec![vec![0.0_f64; n]; bw + 1];
    for k in 0..=bw {
        for j in 0..n.saturating_sub(k) {
            ab[k][j] = a[j + k][j];
        }
    }
    ab
}

/// LAPACK upper-band storage of an SPD matrix A.
/// ab[k][j] = A[j-k][j] for k=0..bw, j=k..n-1. Unused slots set to 0.
fn upper_band_from_dense(a: &[Vec<f64>], bw: usize) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut ab = vec![vec![0.0_f64; n]; bw + 1];
    for k in 0..=bw {
        for j in k..n {
            ab[k][j] = a[j - k][j];
        }
    }
    ab
}

fn build_query() -> OracleQuery {
    let mut pts = Vec::new();

    // 1. Diagonal (bw=0), lower
    {
        let a = vec![
            vec![2.0_f64, 0.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0, 0.0],
            vec![0.0, 0.0, 4.0, 0.0],
            vec![0.0, 0.0, 0.0, 5.0],
        ];
        let ab = lower_band_from_dense(&a, 0);
        pts.push(CasePoint {
            case_id: "diag_4x4_lower".into(),
            ab,
            b: vec![2.0, 6.0, 12.0, 20.0],
            lower: true,
            a_dense: a,
        });
    }

    // 2. Tridiagonal SPD (bw=1), lower storage
    {
        let n = 5;
        let mut a = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            a[i][i] = 4.0;
            if i + 1 < n {
                a[i][i + 1] = -1.0;
                a[i + 1][i] = -1.0;
            }
        }
        let ab = lower_band_from_dense(&a, 1);
        pts.push(CasePoint {
            case_id: "tridiag_5x5_lower".into(),
            ab,
            b: vec![1.0, 1.0, 1.0, 1.0, 1.0],
            lower: true,
            a_dense: a,
        });
    }

    // 3. Tridiagonal SPD (bw=1), upper storage (same A as above)
    {
        let n = 5;
        let mut a = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            a[i][i] = 4.0;
            if i + 1 < n {
                a[i][i + 1] = -1.0;
                a[i + 1][i] = -1.0;
            }
        }
        let ab = upper_band_from_dense(&a, 1);
        pts.push(CasePoint {
            case_id: "tridiag_5x5_upper".into(),
            ab,
            b: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            lower: false,
            a_dense: a,
        });
    }

    // 4. Pentadiagonal SPD (bw=2), lower storage
    {
        let n = 6;
        let mut a = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            a[i][i] = 5.0;
            if i + 1 < n {
                a[i][i + 1] = -1.0;
                a[i + 1][i] = -1.0;
            }
            if i + 2 < n {
                a[i][i + 2] = -0.5;
                a[i + 2][i] = -0.5;
            }
        }
        let ab = lower_band_from_dense(&a, 2);
        pts.push(CasePoint {
            case_id: "pentadiag_6x6_lower".into(),
            ab,
            b: vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0],
            lower: true,
            a_dense: a,
        });
    }

    // 5. Pentadiagonal SPD (bw=2), upper storage (same A)
    {
        let n = 6;
        let mut a = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            a[i][i] = 5.0;
            if i + 1 < n {
                a[i][i + 1] = -1.0;
                a[i + 1][i] = -1.0;
            }
            if i + 2 < n {
                a[i][i + 2] = -0.5;
                a[i + 2][i] = -0.5;
            }
        }
        let ab = upper_band_from_dense(&a, 2);
        pts.push(CasePoint {
            case_id: "pentadiag_6x6_upper".into(),
            ab,
            b: vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            lower: false,
            a_dense: a,
        });
    }

    // 6. Tridiagonal SPD with non-uniform diagonals (bw=1), lower
    {
        let diag = [3.0_f64, 4.0, 5.0, 4.0, 3.0];
        let off = [-0.5_f64, -0.7, -0.3, -0.4];
        let n = diag.len();
        let mut a = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            a[i][i] = diag[i];
            if i + 1 < n {
                a[i][i + 1] = off[i];
                a[i + 1][i] = off[i];
            }
        }
        let ab = lower_band_from_dense(&a, 1);
        pts.push(CasePoint {
            case_id: "tridiag_nonuni_5x5_lower".into(),
            ab,
            b: vec![1.0, 2.0, 0.0, -1.0, 1.0],
            lower: true,
            a_dense: a,
        });
    }

    OracleQuery { points: pts }
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
import numpy as np
from scipy.linalg import solveh_banded

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    cid = c["case_id"]
    try:
        ab = np.array(c["ab"], dtype=float)
        b = np.array(c["b"], dtype=float)
        x = solveh_banded(ab, b, lower=bool(c["lower"]))
        if not np.all(np.isfinite(x)):
            out.append({"case_id": cid, "x": None})
        else:
            out.append({"case_id": cid, "x": [float(v) for v in x]})
    except Exception:
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
            eprintln!("skipping solveh_banded oracle: python3 unavailable ({e})");
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
            eprintln!("skipping solveh_banded oracle: stdin write failed");
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
        eprintln!("skipping solveh_banded oracle: scipy not available\n{stderr}");
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
fn diff_linalg_solveh_banded() {
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

        let sol = match solveh_banded(&case.ab, &case.b, case.lower) {
            Ok(r) => r.x,
            Err(e) => {
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    max_abs_diff: f64::INFINITY,
                    max_rel_diff: f64::INFINITY,
                    residual_inf: f64::INFINITY,
                    pass: false,
                    note: format!("solveh_banded error: {e:?}"),
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

        // Residual against the dense form of A
        let ax = matvec(&case.a_dense, &sol);
        let mut residual_inf = 0.0_f64;
        for (axi, bi) in ax.iter().zip(case.b.iter()) {
            residual_inf = residual_inf.max((axi - bi).abs());
        }

        let close_to_scipy = max_rel <= REL_TOL || max_abs <= ABS_TOL;
        let small_residual = residual_inf <= RESIDUAL_TOL;
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
        test_id: "diff_linalg_solveh_banded".into(),
        category: "fsci_linalg::solveh_banded vs scipy.linalg".into(),
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
                "solveh_banded mismatch: {} rel={} abs={} res={} note={}",
                d.case_id, d.max_rel_diff, d.max_abs_diff, d.residual_inf, d.note
            );
        }
    }

    assert!(
        all_pass,
        "solveh_banded parity failed: {} cases",
        diffs.len()
    );
}
