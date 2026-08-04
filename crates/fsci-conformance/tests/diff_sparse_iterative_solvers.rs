#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_sparse's iterative
//! solvers on well-conditioned SPD/symmetric systems:
//!   - `cg`       vs `scipy.sparse.linalg.cg`
//!   - `gmres`    vs `scipy.sparse.linalg.gmres`
//!   - `bicgstab` vs `scipy.sparse.linalg.bicgstab`
//!   - `minres`   vs `scipy.sparse.linalg.minres`
//!
//! Resolves [frankenscipy-x6tbi]. SPD systems have unique solutions
//! so a vector-element comparison is well-defined. Both implementations
//! solve to ~1e-8 internally; tolerance 1e-6 abs covers the residual
//! floor on probe systems.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{
    CooMatrix, FormatConvertible, IterativeSolveOptions, Shape2D, bicgstab, cg, gmres, lsmr, minres,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-004";
const ABS_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    solver: String,
    n: usize,
    /// COO triplets (row, col, value).
    triplets: Vec<(usize, usize, f64)>,
    b: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Solution vector.
    x: Option<Vec<f64>>,
    /// Why the oracle produced no solution, when it produced none.
    #[serde(default)]
    why: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    solver: String,
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
    fs::create_dir_all(output_dir()).expect("create iter_solvers diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize iter_solvers diff log");
    fs::write(path, json).expect("write iter_solvers diff log");
}

/// Build a small SPD system from triplets + rhs.
fn system_4x4_tridiag_spd() -> (Vec<(usize, usize, f64)>, Vec<f64>) {
    // A = [[4,1,0,0],[1,4,1,0],[0,1,4,1],[0,0,1,4]]
    let trips = vec![
        (0, 0, 4.0),
        (0, 1, 1.0),
        (1, 0, 1.0),
        (1, 1, 4.0),
        (1, 2, 1.0),
        (2, 1, 1.0),
        (2, 2, 4.0),
        (2, 3, 1.0),
        (3, 2, 1.0),
        (3, 3, 4.0),
    ];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    (trips, b)
}

fn system_5x5_diag_spd() -> (Vec<(usize, usize, f64)>, Vec<f64>) {
    // Strict diagonal dominance: A = diag(2,3,4,5,6)
    let trips = (0..5).map(|i| (i, i, (i as f64) + 2.0)).collect();
    let b = vec![1.0, -1.0, 2.0, 3.0, 0.5];
    (trips, b)
}

fn system_6x6_pentadiag_spd() -> (Vec<(usize, usize, f64)>, Vec<f64>) {
    // Symmetric pentadiagonal SPD: A = diag*5 + offdiag*1
    let n = 6;
    let mut trips = Vec::new();
    for i in 0..n {
        trips.push((i, i, 5.0));
        if i + 1 < n {
            trips.push((i, i + 1, 1.0));
            trips.push((i + 1, i, 1.0));
        }
        if i + 2 < n {
            trips.push((i, i + 2, 0.5));
            trips.push((i + 2, i, 0.5));
        }
    }
    let b = (1..=n).map(|i| i as f64).collect();
    (trips, b)
}

/// Symmetric **indefinite** system: the 8x8 Dirichlet five-point Laplacian
/// shifted by 3.7, which puts 28 of its 64 eigenvalues below zero.
///
/// Every other system here is SPD, so before this case MINRES was only ever
/// exercised on operators CG could have handled — the one matrix class it
/// exists for went untested, which is how a GMRES delegate passed this harness.
/// CG and BiCGSTAB are deliberately not run on it: CG has no contract on an
/// indefinite operator.
fn system_64x64_shifted_laplacian_indefinite() -> (Vec<(usize, usize, f64)>, Vec<f64>) {
    const SIDE: usize = 8;
    const DIAGONAL: f64 = 4.001 - 3.7;
    let n = SIDE * SIDE;
    let mut trips = Vec::new();
    for row in 0..SIDE {
        for col in 0..SIDE {
            let index = row * SIDE + col;
            if row > 0 {
                trips.push((index, index - SIDE, -1.0));
            }
            if col > 0 {
                trips.push((index, index - 1, -1.0));
            }
            trips.push((index, index, DIAGONAL));
            if col + 1 < SIDE {
                trips.push((index, index + 1, -1.0));
            }
            if row + 1 < SIDE {
                trips.push((index, index + SIDE, -1.0));
            }
        }
    }
    let b = (0..n).map(|i| 1.0 + 0.1 * (i % 7) as f64).collect();
    (trips, b)
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    let systems: &[(&str, fn() -> (Vec<(usize, usize, f64)>, Vec<f64>), usize)] = &[
        ("4x4_tridiag_spd", system_4x4_tridiag_spd, 4),
        ("5x5_diag_spd", system_5x5_diag_spd, 5),
        ("6x6_pentadiag_spd", system_6x6_pentadiag_spd, 6),
    ];
    let solvers = ["cg", "gmres", "bicgstab", "minres", "lsmr"];
    for (label, fac, n) in systems {
        let (trips, b) = fac();
        for solver in solvers {
            points.push(PointCase {
                case_id: format!("{solver}_{label}"),
                solver: solver.into(),
                n: *n,
                triplets: trips.clone(),
                b: b.clone(),
            });
        }
    }

    let (indefinite_trips, indefinite_b) = system_64x64_shifted_laplacian_indefinite();
    for solver in ["minres", "gmres"] {
        points.push(PointCase {
            case_id: format!("{solver}_64x64_shifted_laplacian_indefinite"),
            solver: solver.into(),
            n: 64,
            triplets: indefinite_trips.clone(),
            b: indefinite_b.clone(),
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
import scipy.sparse as sp
import scipy.sparse.linalg as spl

def finite_vec_or_none(arr):
    out = []
    for v in np.asarray(arr).tolist():
        try:
            v = float(v)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out

SOLVERS = {
    "cg":       spl.cg,
    "gmres":    spl.gmres,
    "bicgstab": spl.bicgstab,
    "minres":   spl.minres,
    "lsmr":     spl.lsmr,
}

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; solver = case["solver"]
    n = case["n"]
    triplets = case["triplets"]
    if triplets:
        r = np.array([t[0] for t in triplets], dtype=int)
        c = np.array([t[1] for t in triplets], dtype=int)
        v = np.array([t[2] for t in triplets], dtype=float)
    else:
        r = np.zeros(0, dtype=int); c = np.zeros(0, dtype=int); v = np.zeros(0, dtype=float)
    b = np.array(case["b"], dtype=float)
    try:
        A = sp.csr_matrix((v, (r, c)), shape=(n, n))
        fn = SOLVERS[solver]
        if solver == "lsmr":
            x, istop, *_ = fn(A, b, atol=1e-10, btol=1e-10, maxiter=500)
            if istop in (0, 1, 2):
                points.append({"case_id": cid, "x": finite_vec_or_none(x)})
            else:
                points.append({"case_id": cid, "x": None, "why": f"istop={istop}"})
            continue
        # `minres` has no `atol` parameter (signature is rtol/shift/maxiter/M).
        # Passing one raises TypeError, which this block used to swallow into a
        # null arm — so every minres case was silently skipped and the minres
        # column of this harness tested nothing at all.
        kwargs = {"rtol": 1e-10, "maxiter": 500}
        if solver != "minres":
            kwargs["atol"] = 0.0
        x, info = fn(A, b, **kwargs)
        if info == 0:
            points.append({"case_id": cid, "x": finite_vec_or_none(x)})
        else:
            points.append({"case_id": cid, "x": None, "why": f"info={info}"})
    except Exception as exc:
        points.append({"case_id": cid, "x": None, "why": f"{type(exc).__name__}: {exc}"})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize iter_solvers query");
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
                "failed to spawn python3 for iter_solvers oracle: {e}"
            );
            eprintln!("skipping iter_solvers oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open iter_solvers oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "iter_solvers oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping iter_solvers oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for iter_solvers oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "iter_solvers oracle failed: {stderr}"
        );
        eprintln!("skipping iter_solvers oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse iter_solvers oracle JSON"))
}

fn fsci_solve(case: &PointCase) -> Option<Vec<f64>> {
    let r: Vec<usize> = case.triplets.iter().map(|t| t.0).collect();
    let c: Vec<usize> = case.triplets.iter().map(|t| t.1).collect();
    let d: Vec<f64> = case.triplets.iter().map(|t| t.2).collect();
    let coo = CooMatrix::from_triplets(Shape2D::new(case.n, case.n), d, r, c, false).ok()?;
    let csr = coo.to_csr().ok()?;
    let opts = IterativeSolveOptions {
        tol: 1.0e-10,
        max_iter: Some(500),
        ..Default::default()
    };
    let result = match case.solver.as_str() {
        "cg" => cg(&csr, &case.b, None, opts).ok()?,
        "gmres" => gmres(&csr, &case.b, None, opts).ok()?,
        "bicgstab" => bicgstab(&csr, &case.b, None, opts).ok()?,
        "minres" => minres(&csr, &case.b, None, opts).ok()?,
        "lsmr" => lsmr(&csr, &case.b, opts).ok()?,
        _ => return None,
    };
    if !result.converged {
        return None;
    }
    Some(result.solution)
}

#[test]
fn diff_sparse_iterative_solvers() {
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

    // A case that neither arm can solve used to `continue` silently, so a
    // harness that compared nothing still reported "pass". Skips are now
    // recorded and asserted on below.
    let mut skipped: Vec<String> = Vec::new();

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_x) = scipy_arm.x.as_ref() else {
            skipped.push(format!(
                "{} (scipy arm: {})",
                case.case_id,
                scipy_arm.why.as_deref().unwrap_or("no solution")
            ));
            continue;
        };
        let Some(fsci_x) = fsci_solve(case) else {
            skipped.push(format!("{} (fsci arm did not converge)", case.case_id));
            continue;
        };
        if fsci_x.len() != scipy_x.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                solver: case.solver.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let abs_d = fsci_x
            .iter()
            .zip(scipy_x.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            solver: case.solver.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    // Every solver named in the query must have been genuinely compared at
    // least once. Without this, a solver whose oracle call raises — as
    // `minres` did for years, on an unsupported `atol` kwarg — contributes
    // zero cases and the harness reports a green "pass" over an empty column.
    for solver in ["cg", "gmres", "bicgstab", "minres", "lsmr"] {
        let compared = diffs.iter().filter(|d| d.solver == solver).count();
        assert!(
            compared > 0,
            "iter_solvers compared zero cases for `{solver}`; this harness cannot \
             pass by testing nothing. Skipped: {skipped:?}"
        );
    }
    if !skipped.is_empty() {
        eprintln!(
            "iter_solvers skipped {} case(s): {skipped:?}",
            skipped.len()
        );
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_iterative_solvers".into(),
        category: "scipy.sparse.linalg cg/gmres/bicgstab/minres".into(),
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
            eprintln!(
                "iter_solvers {} mismatch: {} abs_diff={}",
                d.solver, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.sparse iterative-solver conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
