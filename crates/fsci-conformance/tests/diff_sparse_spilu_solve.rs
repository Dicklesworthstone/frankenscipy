#![forbid(unsafe_code)]
//! Cover fsci_sparse::spilu factorization quality.
//!
//! Resolves [frankenscipy-sa6bh]. ILU is approximate, so direct
//! bit-for-bit parity against scipy.sparse.linalg.spilu is not the
//! right invariant. We instead verify:
//!   * On diagonal matrices, ILU has zero fill-in → solve is exact.
//!   * On tridiagonal SPD matrices, ILU has zero fill-in → solve is
//!     exact to machine precision.
//!   * On moderate-fill banded matrices, ||A x − b||∞ is below an
//!     ILU-realistic bound (matches scipy.sparse.linalg.spilu's typical
//!     residual quality after default drop_tol).
//!
//! Each probe also runs scipy spilu(A).solve(b) as a sanity baseline.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CooMatrix, FormatConvertible, IluOptions, Shape2D, spilu};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    /// Triplets (row, col, value)
    triplets: Vec<(usize, usize, f64)>,
    rows: usize,
    cols: usize,
    b: Vec<f64>,
    /// Per-case residual tolerance on ||A x − b||∞
    residual_tol: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<CasePoint>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OraclePoint {
    case_id: String,
    /// scipy's residual ||A x_scipy − b||∞ for comparison context
    scipy_residual_inf: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OraclePoint>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    fsci_residual_inf: f64,
    scipy_residual_inf: Option<f64>,
    residual_tol: f64,
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
    fs::create_dir_all(output_dir()).expect("create spilu diff dir");
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

    // 1. Diagonal 5x5: ILU exact (no fill-in possible)
    pts.push(CasePoint {
        case_id: "diag_5x5".into(),
        triplets: (0..5).map(|i| (i, i, (i + 2) as f64)).collect(),
        rows: 5,
        cols: 5,
        b: vec![2.0, 6.0, 12.0, 20.0, 30.0],
        residual_tol: 1.0e-12,
    });

    // 2. Tridiagonal SPD 6x6: ILU also exact (no fill-in)
    {
        let n = 6;
        let mut trips = Vec::new();
        for i in 0..n {
            trips.push((i, i, 4.0));
            if i + 1 < n {
                trips.push((i, i + 1, -1.0));
                trips.push((i + 1, i, -1.0));
            }
        }
        pts.push(CasePoint {
            case_id: "tridiag_6x6_spd".into(),
            triplets: trips,
            rows: n,
            cols: n,
            b: vec![1.0; n],
            residual_tol: 1.0e-10,
        });
    }

    // 3. Tridiagonal nonsymmetric 6x6: still exact ILU
    {
        let n = 6;
        let mut trips = Vec::new();
        for i in 0..n {
            trips.push((i, i, 4.0));
            if i + 1 < n {
                trips.push((i, i + 1, -2.0));
                trips.push((i + 1, i, -1.0));
            }
        }
        pts.push(CasePoint {
            case_id: "tridiag_6x6_nonsym".into(),
            triplets: trips,
            rows: n,
            cols: n,
            b: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            residual_tol: 1.0e-10,
        });
    }

    // 4. Pentadiagonal SPD 8x8 (still well-conditioned, small fill-in)
    {
        let n = 8;
        let mut trips = Vec::new();
        for i in 0..n {
            trips.push((i, i, 5.0));
            if i + 1 < n {
                trips.push((i, i + 1, -1.0));
                trips.push((i + 1, i, -1.0));
            }
            if i + 2 < n {
                trips.push((i, i + 2, -0.5));
                trips.push((i + 2, i, -0.5));
            }
        }
        pts.push(CasePoint {
            case_id: "pentadiag_8x8_spd".into(),
            triplets: trips,
            rows: n,
            cols: n,
            b: (0..n).map(|i| i as f64).collect(),
            // ILU on pentadiagonal SPD with default drop_tol=1e-4 produces
            // residuals around 1e-3 to 1e-4 depending on conditioning.
            residual_tol: 1.0e-3,
        });
    }

    OracleQuery { points: pts }
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    cid = c["case_id"]
    try:
        rows = np.array([t[0] for t in c["triplets"]], dtype=int)
        cols = np.array([t[1] for t in c["triplets"]], dtype=int)
        data = np.array([t[2] for t in c["triplets"]], dtype=float)
        A_coo = sp.coo_matrix((data, (rows, cols)), shape=(c["rows"], c["cols"]))
        A_csc = A_coo.tocsc()
        b = np.array(c["b"], dtype=float)
        ilu = spl.spilu(A_csc)
        x = ilu.solve(b)
        r = A_csc @ x - b
        rinf = float(np.max(np.abs(r))) if r.size else 0.0
        out.append({"case_id": cid, "scipy_residual_inf": rinf})
    except Exception:
        out.append({"case_id": cid, "scipy_residual_inf": None})

print(json.dumps({"points": out}))
"#;
    let query_json = serde_json::to_string(q).expect("serialize");
    let mut child = match Command::new("python3")
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
            eprintln!("skipping spilu oracle: python3 unavailable ({e})");
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
            eprintln!("skipping spilu oracle: stdin write failed");
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
        eprintln!("skipping spilu oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

/// Multiply a COO-like triplet list by a dense vector to get residual.
fn matvec_triplets(triplets: &[(usize, usize, f64)], x: &[f64], rows: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; rows];
    for &(r, c, v) in triplets {
        out[r] += v * x[c];
    }
    out
}

#[test]
fn diff_sparse_spilu_solve() {
    let query = build_query();
    let oracle = scipy_oracle_or_skip(&query);

    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    for (i, case) in query.points.iter().enumerate() {
        let scipy_res = oracle
            .as_ref()
            .and_then(|o| o.points.get(i))
            .and_then(|p| p.scipy_residual_inf);

        let data: Vec<f64> = case.triplets.iter().map(|t| t.2).collect();
        let rs: Vec<usize> = case.triplets.iter().map(|t| t.0).collect();
        let cs: Vec<usize> = case.triplets.iter().map(|t| t.1).collect();
        let coo = match CooMatrix::from_triplets(
            Shape2D::new(case.rows, case.cols),
            data,
            rs,
            cs,
            true,
        ) {
            Ok(m) => m,
            Err(e) => {
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    fsci_residual_inf: f64::INFINITY,
                    scipy_residual_inf: scipy_res,
                    residual_tol: case.residual_tol,
                    pass: false,
                    note: format!("COO build error: {e:?}"),
                });
                continue;
            }
        };
        let csc = match coo.to_csc() {
            Ok(m) => m,
            Err(e) => {
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    fsci_residual_inf: f64::INFINITY,
                    scipy_residual_inf: scipy_res,
                    residual_tol: case.residual_tol,
                    pass: false,
                    note: format!("to_csc error: {e:?}"),
                });
                continue;
            }
        };
        let ilu = match spilu(&csc, IluOptions::default()) {
            Ok(f) => f,
            Err(e) => {
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    fsci_residual_inf: f64::INFINITY,
                    scipy_residual_inf: scipy_res,
                    residual_tol: case.residual_tol,
                    pass: false,
                    note: format!("spilu error: {e:?}"),
                });
                continue;
            }
        };
        let x = match ilu.solve(&case.b) {
            Ok(v) => v,
            Err(e) => {
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    fsci_residual_inf: f64::INFINITY,
                    scipy_residual_inf: scipy_res,
                    residual_tol: case.residual_tol,
                    pass: false,
                    note: format!("ilu.solve error: {e:?}"),
                });
                continue;
            }
        };

        // Compute residual A x − b
        let ax = matvec_triplets(&case.triplets, &x, case.rows);
        let mut residual_inf = 0.0_f64;
        for (axi, bi) in ax.iter().zip(case.b.iter()) {
            residual_inf = residual_inf.max((axi - bi).abs());
        }

        let pass = residual_inf <= case.residual_tol;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            fsci_residual_inf: residual_inf,
            scipy_residual_inf: scipy_res,
            residual_tol: case.residual_tol,
            pass,
            note: String::new(),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_sparse_spilu_solve".into(),
        category: "fsci_sparse::spilu factorization + .solve(b) residual quality".into(),
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
                "spilu mismatch: {} fsci_res={} scipy_res={:?} tol={} note={}",
                d.case_id, d.fsci_residual_inf, d.scipy_residual_inf, d.residual_tol, d.note
            );
        }
    }

    assert!(
        all_pass,
        "spilu residual coverage failed: {} cases",
        diffs.len()
    );
}
