#![forbid(unsafe_code)]
//! Live scipy.sparse.linalg parity for `fsci_sparse::lobpcg` (frankenscipy-6j5tz).
//!
//! fsci's `lobpcg` transcribes SciPy's `lobpcg.py`, and both start from the same block `X`
//! (sent explicitly), so they run the same LOBPCG recurrence and differ by rounding (small dense
//! eigenproblems through nalgebra instead of LAPACK, unblocked products instead of BLAS). Arms:
//! - `eigenvalues`: difference relative to the largest returned eigenvalue magnitude ≤ 1e-8 (an
//!   eigenvalue at 0, a Neumann Laplacian's, is only determined to the operator's rounding).
//! - `eigenvectors`: for each returned vector, the sine of its angle to the span of SciPy's
//!   vectors for the same eigenvalue (a cluster at the eigenvalue tolerance) ≤ 1e-8.
//!
//! Each case also measures SciPy against ITSELF from a start block moved by one ulp. Where that
//! movement exceeds the bounds above (a run that stagnates without converging amplifies rounding:
//! `diag80_largest4_tol1e-10_precond` moves 4.5e-7 under one ulp, while a converging run moves
//! ~1e-16), the case's bound is ten times SciPy's own movement. It never loosens a stable case,
//! and on the stagnating one the fsci–SciPy difference grows smoothly from 1.1e-15 at iteration 0
//! (its histories are printed on any mismatch), the signature of rounding, not of another path.
//! - `converged`: whether every final residual is within `tol` (SciPy warns otherwise), exactly.
//! - `history`: the number of rows of `lambdaHistory`/`residualNormsHistory` SciPy returns,
//!   which encodes the best iteration it kept, exactly (none for the dense branch on either side).
//!
//! The 1e-8 bounds are deliberately far below the convergence tolerances (`√ε·n` ≈ 1e-6): a
//! recurrence that converged along another path would land 1e-4..1e-6 away in the vectors, so
//! agreement at 1e-8 is evidence of the same path, and a non-converged case (`maxiter` hit)
//! compares the same partial iterate. Cases: standard, generalized (`B`), preconditioned (`M`),
//! constrained (`Y`), largest and smallest, matrix-free operators (a closure-defined 1-D
//! Laplacian and SciPy's `-LaplacianNd`), the dense branch (`n < 5k`) and a run that stops at
//! `maxiter`.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_sparse::{
    CooMatrix, CsrMatrix, FormatConvertible, FunctionOperator, LaplacianBoundary, LaplacianNd,
    LinearOperator, LobpcgOptions, ScaledOperator, Shape2D, lobpcg,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const EIGENVALUE_TOL: f64 = 1.0e-8;
const SUBSPACE_TOL: f64 = 1.0e-8;
/// How far beyond SciPy's own one-ulp sensitivity a case may move.
const SENSITIVITY_FACTOR: f64 = 10.0;
const ARMS: [&str; 4] = ["eigenvalues", "eigenvectors", "converged", "history"];

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind")]
enum OpSpec {
    /// `diag(values)` as a sparse matrix.
    #[serde(rename = "diag")]
    Diag { values: Vec<f64> },
    /// Symmetric tridiagonal `(off, diag)` as a sparse matrix.
    #[serde(rename = "tridiag")]
    Tridiag { n: usize, off: f64, diag: f64 },
    /// The 1-D Dirichlet Laplacian `2x_i − x_{i−1} − x_{i+1}` applied by a closure.
    #[serde(rename = "lap1d")]
    Lap1d { n: usize },
    /// `−LaplacianNd(grid, boundary_conditions=bc)`.
    #[serde(rename = "neg_lapnd")]
    NegLapNd { grid: Vec<usize>, bc: String },
}

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    n: usize,
    a: OpSpec,
    b: Option<OpSpec>,
    /// The preconditioner `diag(1 / values)`.
    m_inverse_diag: Option<Vec<f64>>,
    y: Option<Vec<Vec<f64>>>,
    x: Vec<Vec<f64>>,
    tol: Option<f64>,
    maxiter: usize,
    largest: bool,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    cases: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleCase {
    case_id: String,
    eigenvalues: Option<Vec<f64>>,
    eigenvectors: Option<Vec<Vec<f64>>>,
    converged: Option<bool>,
    history_rows: Option<usize>,
    #[serde(default)]
    lambda_history: Option<Vec<Vec<f64>>>,
    /// SciPy against itself from a start block moved by one ulp: eigenvalues, relative to the
    /// largest returned magnitude.
    eig_sensitivity: Option<f64>,
    /// The same for the eigenvectors (largest sine to the matching span).
    vec_sensitivity: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    cases: Vec<OracleCase>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    arm: String,
    metric: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    max_metric: BTreeMap<String, f64>,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn emit_log(log: &DiffLog) {
    fs::create_dir_all(output_dir()).expect("create diff dir");
    let path = output_dir().join(format!("{}.json", log.test_id));
    fs::write(path, serde_json::to_string_pretty(log).expect("serialize")).expect("write log");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

/// Uniform(−1, 1) start block from a 64-bit LCG.
fn start_block(n: usize, k: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    (0..k)
        .map(|_| {
            (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
                })
                .collect()
        })
        .collect()
}

#[allow(clippy::too_many_lines)]
fn generate_query() -> OracleQuery {
    let mut cases = Vec::new();
    let mut add =
        |case_id: &str, a: OpSpec, n: usize, k: usize, seed: u64, configure: &dyn Fn(&mut Case)| {
            let mut case = Case {
                case_id: case_id.to_string(),
                n,
                a,
                b: None,
                m_inverse_diag: None,
                y: None,
                x: start_block(n, k, seed),
                tol: None,
                maxiter: 100,
                largest: true,
            };
            configure(&mut case);
            cases.push(case);
        };
    let ramp = |n: usize| (1..=n).map(|v| v as f64).collect::<Vec<f64>>();

    add(
        "diag100_largest3",
        OpSpec::Diag { values: ramp(100) },
        100,
        3,
        1,
        &|_| {},
    );
    add(
        "diag100_smallest3_precond",
        OpSpec::Diag { values: ramp(100) },
        100,
        3,
        2,
        &|c| {
            c.largest = false;
            c.m_inverse_diag = Some((1..=100).map(|v| v as f64).collect());
        },
    );
    add(
        "diag80_largest4_tol1e-10_precond",
        OpSpec::Diag {
            values: (0..80).map(|i| 1.0 + (i as f64).powf(1.3)).collect(),
        },
        80,
        4,
        3,
        &|c| {
            c.tol = Some(1e-10);
            c.m_inverse_diag = Some((0..80).map(|i| 1.0 + (i as f64).powf(1.3)).collect());
        },
    );
    add(
        "lap1d60_largest4",
        OpSpec::Lap1d { n: 60 },
        60,
        4,
        0x9E37_79B9_7F4A_7C15,
        &|c| {
            c.maxiter = 300;
            c.tol = Some(1e-9);
        },
    );
    add(
        "lap1d200_smallest3_maxiter3",
        OpSpec::Lap1d { n: 200 },
        200,
        3,
        0x9E37_79B9_7F4A_7C15,
        &|c| {
            c.largest = false;
            c.maxiter = 3;
            c.tol = Some(1e-12);
        },
    );
    add(
        "lap1d120_largest2_maxiter10",
        OpSpec::Lap1d { n: 120 },
        120,
        2,
        7,
        &|c| {
            c.maxiter = 10;
        },
    );
    add(
        "neg_lapnd_10x8_dirichlet_largest3",
        OpSpec::NegLapNd {
            grid: vec![10, 8],
            bc: "dirichlet".into(),
        },
        80,
        3,
        4,
        &|c| c.maxiter = 200,
    );
    add(
        "neg_lapnd_9x7_neumann_smallest2_maxiter40",
        OpSpec::NegLapNd {
            grid: vec![9, 7],
            bc: "neumann".into(),
        },
        63,
        2,
        5,
        &|c| {
            c.largest = false;
            c.maxiter = 40;
        },
    );
    add(
        "tridiag_generalized_smallest3",
        OpSpec::Tridiag {
            n: 90,
            off: -1.0,
            diag: 2.5,
        },
        90,
        3,
        6,
        &|c| {
            c.largest = false;
            c.b = Some(OpSpec::Diag {
                values: (0..90).map(|i| 1.0 + i as f64 / 90.0).collect(),
            });
            c.maxiter = 200;
        },
    );
    add(
        "tridiag_generalized_largest2_precond",
        OpSpec::Tridiag {
            n: 70,
            off: -1.0,
            diag: 3.0,
        },
        70,
        2,
        8,
        &|c| {
            c.b = Some(OpSpec::Diag {
                values: (0..70).map(|i| 2.0 - i as f64 / 70.0).collect(),
            });
            c.m_inverse_diag = Some(vec![3.0; 70]);
            c.maxiter = 200;
        },
    );
    add(
        "diag60_constrained_largest2",
        OpSpec::Diag { values: ramp(60) },
        60,
        2,
        9,
        &|c| {
            let mut top = vec![0.0; 60];
            top[59] = 1.0;
            let mut next = vec![0.0; 60];
            next[58] = 1.0;
            c.y = Some(vec![top, next]);
        },
    );
    add(
        "dense_branch_12_k3",
        OpSpec::Diag {
            values: (0..12).map(|i| (i as f64 - 5.5).powi(2)).collect(),
        },
        12,
        3,
        10,
        &|_| {},
    );
    add(
        "dense_branch_tridiag_9_k2_smallest",
        OpSpec::Tridiag {
            n: 9,
            off: -1.0,
            diag: 2.0,
        },
        9,
        2,
        11,
        &|c| {
            c.largest = false;
        },
    );
    OracleQuery { cases }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LaplacianNd, LinearOperator, lobpcg

def build(spec, n):
    kind = spec["kind"]
    if kind == "diag":
        return sp.diags_array(np.array(spec["values"], dtype=float)).tocsr()
    if kind == "tridiag":
        m = int(spec["n"])
        off, d = float(spec["off"]), float(spec["diag"])
        return sp.diags_array([np.full(m - 1, off), np.full(m, d), np.full(m - 1, off)],
                              offsets=[-1, 0, 1]).tocsr()
    if kind == "lap1d":
        m = int(spec["n"])
        def mm(x):
            x = np.asarray(x, dtype=float)
            y = 2.0 * x
            y[1:] -= x[:-1]
            y[:-1] -= x[1:]
            return y
        return LinearOperator((m, m), matvec=mm, matmat=mm, dtype=float)
    if kind == "neg_lapnd":
        return -LaplacianNd(tuple(spec["grid"]), boundary_conditions=spec["bc"], dtype=np.float64)
    raise ValueError(kind)

def sine_to_span(v, basis):
    q, _ = np.linalg.qr(np.array(basis, dtype=float).T)
    v = np.asarray(v, dtype=float) / np.linalg.norm(v)
    return float(np.linalg.norm(v - q @ (q.T @ v)))

q = json.load(sys.stdin)
out = []
for case in q["cases"]:
    cid = case["case_id"]
    n = int(case["n"])
    row = {"case_id": cid, "eigenvalues": None, "eigenvectors": None, "converged": None,
           "history_rows": None, "eig_sensitivity": None, "vec_sensitivity": None}
    try:
        A = build(case["a"], n)
        B = None if case["b"] is None else build(case["b"], n)
        M = None
        if case["m_inverse_diag"] is not None:
            M = sp.diags_array(1.0 / np.array(case["m_inverse_diag"], dtype=float)).tocsr()
        Y = None if case["y"] is None else np.array(case["y"], dtype=float).T
        X = np.array(case["x"], dtype=float).T
        tol = case["tol"]
        def run(X0):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return lobpcg(A, X0, B=B, M=M, Y=Y, tol=tol, maxiter=case["maxiter"],
                              largest=case["largest"], retLambdaHistory=True,
                              retResidualNormsHistory=True)
        res = run(X)
        # SciPy's own sensitivity: the same call from a start block moved by one ulp.
        alt = run(np.nextafter(X, np.inf))
        lam_a, lam_b = np.asarray(res[0]), np.asarray(alt[0])
        scale = max(float(np.max(np.abs(lam_a))), 1e-300)
        row["eig_sensitivity"] = float(np.max(np.abs(lam_a - lam_b)) / scale)
        Va, Vb = np.asarray(res[1]), np.asarray(alt[1])
        row["vec_sensitivity"] = max(
            sine_to_span(Va[:, j], [Vb[:, i] for i in range(len(lam_b))
                                    if abs(lam_b[i] - lam_a[j]) <= 1e-6 * scale] or [Vb[:, j]])
            for j in range(len(lam_a)))
        if len(res) == 4:
            lam, V, lh, rh = res
            used_tol = tol if (tol is not None and tol > 0) else np.sqrt(np.finfo(float).eps) * n
            row["converged"] = bool(np.max(np.abs(rh[-1])) <= used_tol)
            row["history_rows"] = len(lh)
            row["lambda_history"] = [[float(v) for v in np.atleast_1d(r)] for r in lh]
        else:
            lam, V = res  # the dense branch returns no history
            row["converged"] = True
            row["history_rows"] = 0
        row["eigenvalues"] = [float(v) for v in lam]
        row["eigenvectors"] = [[float(v) for v in col] for col in np.asarray(V).T]
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e!r}\n")
    out.append(row)
print(json.dumps({"cases": out}))
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
                "failed to spawn python for lobpcg oracle: {e}"
            );
            eprintln!("skipping lobpcg oracle: python not available ({e})");
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
                "lobpcg oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping lobpcg oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "lobpcg oracle failed: {stderr}"
        );
        eprintln!("skipping lobpcg oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse oracle JSON"))
}

fn csr_from(n: usize, entries: &[(usize, usize, f64)]) -> CsrMatrix {
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for &(i, j, x) in entries {
        r.push(i);
        c.push(j);
        v.push(x);
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), v, r, c, true)
        .and_then(|coo| coo.to_csr())
        .expect("csr")
}

fn build(spec: &OpSpec, n: usize) -> Box<dyn LinearOperator> {
    match spec {
        OpSpec::Diag { values } => {
            let entries: Vec<_> = values.iter().enumerate().map(|(i, &v)| (i, i, v)).collect();
            Box::new(csr_from(n, &entries))
        }
        OpSpec::Tridiag { n, off, diag } => {
            let mut entries = Vec::new();
            for i in 0..*n {
                entries.push((i, i, *diag));
                if i + 1 < *n {
                    entries.push((i, i + 1, *off));
                    entries.push((i + 1, i, *off));
                }
            }
            Box::new(csr_from(*n, &entries))
        }
        OpSpec::Lap1d { n } => Box::new(FunctionOperator::new(
            Shape2D::new(*n, *n),
            |x: &[f64]| {
                let m = x.len();
                let mut y: Vec<f64> = x.iter().map(|v| 2.0 * v).collect();
                for i in 1..m {
                    y[i] -= x[i - 1];
                }
                for i in 0..m.saturating_sub(1) {
                    y[i] -= x[i + 1];
                }
                Ok(y)
            },
        )),
        OpSpec::NegLapNd { grid, bc } => {
            let boundary: LaplacianBoundary = bc.parse().expect("boundary");
            let lap = LaplacianNd::new(grid, boundary).expect("LaplacianNd");
            Box::new(ScaledOperator::new(lap, -1.0))
        }
    }
}

struct FsciRun {
    eigenvalues: Vec<f64>,
    eigenvectors: Vec<Vec<f64>>,
    converged: bool,
    history_rows: usize,
    lambda_history: Vec<Vec<f64>>,
}

fn fsci_lobpcg(case: &Case) -> Result<FsciRun, String> {
    let n = case.n;
    let a = build(&case.a, n);
    let b = case.b.as_ref().map(|spec| build(spec, n));
    let m = case.m_inverse_diag.as_ref().map(|values| {
        let entries: Vec<_> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, i, 1.0 / v))
            .collect();
        csr_from(n, &entries)
    });
    let result = lobpcg(
        a.as_ref(),
        &case.x,
        b.as_deref(),
        m.as_ref().map(|m| m as &dyn LinearOperator),
        case.y.as_deref(),
        LobpcgOptions {
            tol: case.tol,
            max_iter: case.maxiter,
            largest: case.largest,
            ..LobpcgOptions::default()
        },
    )
    .map_err(|e| e.to_string())?;
    Ok(FsciRun {
        history_rows: if result.dense_fallback {
            0
        } else {
            result.lambda_history.len()
        },
        lambda_history: result.lambda_history,
        eigenvalues: result.eigenvalues,
        eigenvectors: result.eigenvectors,
        converged: result.converged,
    })
}

/// sin of the angle between `v` and `span(basis)` (Gram–Schmidt on the basis).
fn sine_to_span(v: &[f64], basis: &[&Vec<f64>]) -> f64 {
    let mut q: Vec<Vec<f64>> = Vec::new();
    for b in basis {
        let mut w = (*b).clone();
        for _ in 0..2 {
            for qi in &q {
                let c: f64 = qi.iter().zip(&w).map(|(a, b)| a * b).sum();
                for (wi, qv) in w.iter_mut().zip(qi) {
                    *wi -= c * qv;
                }
            }
        }
        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            q.push(w.iter().map(|x| x / norm).collect());
        }
    }
    let v_norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut r: Vec<f64> = v.iter().map(|x| x / v_norm).collect();
    for _ in 0..2 {
        for qi in &q {
            let c: f64 = qi.iter().zip(&r).map(|(a, b)| a * b).sum();
            for (ri, qv) in r.iter_mut().zip(qi) {
                *ri -= c * qv;
            }
        }
    }
    r.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
fn diff_sparse_lobpcg() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    let by_id: HashMap<String, OracleCase> = oracle
        .cases
        .into_iter()
        .map(|c| (c.case_id.clone(), c))
        .collect();

    let start = Instant::now();
    let mut ledger = CompareLedger::new("diff_sparse_lobpcg", &ARMS);
    let mut diffs = Vec::new();
    let mut max_metric: BTreeMap<String, f64> = BTreeMap::new();
    let mut record = |ledger: &mut CompareLedger, case_id: &str, arm: &str, metric: f64| {
        let pass = metric <= 1.0;
        ledger.compared(arm, case_id, pass);
        let worst = max_metric.entry(arm.to_string()).or_insert(0.0);
        *worst = worst.max(metric);
        diffs.push(CaseDiff {
            case_id: case_id.to_string(),
            arm: arm.to_string(),
            metric,
            pass,
        });
    };

    for case in &query.cases {
        let id = case.case_id.as_str();
        let scipy = by_id.get(id);
        let fsci = fsci_lobpcg(case);
        if let Err(e) = &fsci {
            eprintln!("lobpcg fsci error on {id}: {e}");
        }
        let fsci = fsci.ok();

        // The case's tolerances: the same-path bounds, unless SciPy itself moves further than
        // that under a one-ulp change of its start block (a stagnating run amplifies rounding),
        // in which case ten times SciPy's own movement.
        let eigenvalue_tol = scipy
            .and_then(|s| s.eig_sensitivity)
            .map_or(EIGENVALUE_TOL, |sens| {
                EIGENVALUE_TOL.max(SENSITIVITY_FACTOR * sens)
            });
        let subspace_tol = scipy
            .and_then(|s| s.vec_sensitivity)
            .map_or(SUBSPACE_TOL, |sens| {
                SUBSPACE_TOL.max(SENSITIVITY_FACTOR * sens)
            });

        if let Some((s, f)) = ledger.slices(
            "eigenvalues",
            id,
            scipy.and_then(|s| s.eigenvalues.as_deref()),
            fsci.as_ref().map(|f| f.eigenvalues.as_slice()),
        ) {
            // Relative to the spectrum's scale: an eigenvalue at 0 (a Neumann Laplacian's) is
            // only determined to rounding of the operator, not to its own (zero) magnitude.
            let scale = s
                .iter()
                .fold(0.0_f64, |m, v| m.max(v.abs()))
                .max(f64::MIN_POSITIVE);
            let metric = s
                .iter()
                .zip(f)
                .map(|(a, b)| (a - b).abs() / scale)
                .fold(0.0_f64, f64::max);
            if metric > eigenvalue_tol {
                eprintln!("lobpcg {id}: eigenvalues fsci {f:?} vs SciPy {s:?}");
                // Where the two lambda histories separate, iteration by iteration.
                if let (Some(sh), Some(fr)) =
                    (scipy.and_then(|s| s.lambda_history.as_ref()), fsci.as_ref())
                {
                    for (it, (srow, frow)) in sh.iter().zip(&fr.lambda_history).enumerate() {
                        let d = srow
                            .iter()
                            .zip(frow)
                            .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
                            .fold(0.0_f64, f64::max);
                        eprintln!("lobpcg {id}: history row {it}: max rel diff {d:.3e}");
                    }
                }
            }
            record(&mut ledger, id, "eigenvalues", metric / eigenvalue_tol);
        }

        if let Some((s, f)) = ledger.both(
            "eigenvectors",
            id,
            scipy.and_then(|s| Some((s.eigenvalues.as_ref()?, s.eigenvectors.as_ref()?))),
            fsci.as_ref(),
        ) {
            let (scipy_values, scipy_vectors) = s;
            let scale = scipy_values
                .iter()
                .fold(0.0_f64, |m, v| m.max(v.abs()))
                .max(f64::MIN_POSITIVE);
            let metric = if scipy_vectors.len() == f.eigenvectors.len() {
                f.eigenvalues
                    .iter()
                    .zip(&f.eigenvectors)
                    .map(|(&lambda, v)| {
                        // SciPy's vectors for the same eigenvalue (a cluster at the case's
                        // eigenvalue tolerance, relative to the spectrum's scale).
                        let cluster: Vec<&Vec<f64>> = scipy_values
                            .iter()
                            .zip(scipy_vectors)
                            .filter(|(mu, _)| (*mu - lambda).abs() <= eigenvalue_tol * scale)
                            .map(|(_, u)| u)
                            .collect();
                        if cluster.is_empty() {
                            f64::INFINITY
                        } else {
                            sine_to_span(v, &cluster)
                        }
                    })
                    .fold(0.0_f64, f64::max)
            } else {
                f64::INFINITY
            };
            record(&mut ledger, id, "eigenvectors", metric / subspace_tol);
        }

        if let Some((s, f)) = ledger.both(
            "converged",
            id,
            scipy.and_then(|s| s.converged),
            fsci.as_ref().map(|f| f.converged),
        ) {
            record(
                &mut ledger,
                id,
                "converged",
                if s == f { 0.0 } else { f64::INFINITY },
            );
        }
        if let Some((s, f)) = ledger.both(
            "history",
            id,
            scipy.and_then(|s| s.history_rows),
            fsci.as_ref().map(|f| f.history_rows),
        ) {
            record(
                &mut ledger,
                id,
                "history",
                if s == f { 0.0 } else { f64::INFINITY },
            );
            if s != f {
                eprintln!("lobpcg {id}: history rows fsci {f} vs SciPy {s}");
            }
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_sparse_lobpcg".into(),
        category: "fsci_sparse::lobpcg vs scipy.sparse.linalg.lobpcg".into(),
        case_count: query.cases.len(),
        compared: ledger.counts().clone(),
        max_metric: max_metric.clone(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);
    for d in &diffs {
        eprintln!("lobpcg {} {} metric={:.3e}", d.case_id, d.arm, d.metric);
    }
    eprintln!("lobpcg worst metric per arm (1.0 = tolerance): {max_metric:?}");
    ledger.finish(query.cases.len());
}
