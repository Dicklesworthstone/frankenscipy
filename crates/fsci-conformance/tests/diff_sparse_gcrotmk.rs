#![forbid(unsafe_code)]
//! Live scipy.sparse.linalg parity for `fsci_sparse::gcrotmk` (frankenscipy-6j5tz).
//!
//! fsci's `gcrotmk` is a step-for-step transcription of SciPy's `_gcrotmk.py`, so on every case
//! both run the same GCROT(m, k) recurrence and should differ only by rounding (dot products and
//! norms are not BLAS-ordered, the Hessenberg QR is updated by Givens rotations rather than a
//! materialized `qr_insert`). Arms:
//! - `info`: SciPy's `info`, exactly (0 on convergence, else the outer iterations performed).
//! - `solution`: `‖x_fsci − x_scipy‖∞ / ‖x_scipy‖∞ ≤ 1e-9`. That is far below the solve
//!   tolerance on purpose: a recurrence that merely reached the same tolerance by another path
//!   would differ at `rtol` scale (1e-5 here), so passing at 1e-9 is evidence of the same path.
//! - `recycled_info` / `recycled_solution`: a second solve of a perturbed right-hand side that
//!   reuses the `CU` list the first solve left behind (SciPy's in/out argument), same metrics.
//!
//! Cases cover CSR and matrix-free operators (a closure-defined stencil on both sides: SciPy's
//! `LinearOperator(shape, matvec)` and fsci's `FunctionOperator`), `truncate='oldest'|'smallest'`,
//! small `m`/`k`, `x0`, a Jacobi `M`, `discard_C`, budget exhaustion (`info > 0`) and `b = 0`.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_sparse::{
    CooMatrix, CsrMatrix, FormatConvertible, FunctionOperator, GcrotmkOptions, GcrotmkPair,
    GcrotmkTruncate, LinearOperator, Shape2D, gcrotmk,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const SOLUTION_TOL: f64 = 1.0e-9;
const ARMS: [&str; 4] = ["info", "solution", "recycled_info", "recycled_solution"];

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    n: usize,
    /// Explicit CSR entries, or empty for the matrix-free stencil.
    triplets: Vec<(usize, usize, f64)>,
    /// `(lower, diag, upper)` of a tridiagonal stencil applied by a closure.
    stencil: Option<(f64, f64, f64)>,
    b: Vec<f64>,
    b2: Option<Vec<f64>>,
    x0: Option<Vec<f64>>,
    jacobi: bool,
    rtol: f64,
    atol: f64,
    maxiter: usize,
    m: usize,
    k: Option<usize>,
    truncate: String,
    discard_c: bool,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    cases: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleCase {
    case_id: String,
    x: Option<Vec<f64>>,
    info: Option<usize>,
    x2: Option<Vec<f64>>,
    info2: Option<usize>,
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

/// Convection-diffusion on an `side × side` grid (5-point, upwinded), nonsymmetric.
fn convection_2d(side: usize, peclet: f64) -> Vec<(usize, usize, f64)> {
    let n = side * side;
    let mut t = Vec::new();
    for i in 0..side {
        for j in 0..side {
            let row = i * side + j;
            t.push((row, row, 4.0 + peclet));
            if j > 0 {
                t.push((row, row - 1, -1.0 - peclet));
            }
            if j + 1 < side {
                t.push((row, row + 1, -1.0));
            }
            if i > 0 {
                t.push((row, row - side, -1.0));
            }
            if i + 1 < side {
                t.push((row, row + side, -1.0));
            }
        }
    }
    debug_assert!(t.iter().all(|&(r, c, _)| r < n && c < n));
    t
}

fn rhs(n: usize, phase: f64) -> Vec<f64> {
    (0..n)
        .map(|i| ((i as f64) * 0.37 + phase).sin() + 0.3)
        .collect()
}

#[allow(clippy::too_many_lines)]
fn generate_query() -> OracleQuery {
    let base = |case_id: &str, n: usize| Case {
        case_id: case_id.to_string(),
        n,
        triplets: Vec::new(),
        stencil: None,
        b: rhs(n, 0.0),
        b2: None,
        x0: None,
        jacobi: false,
        rtol: 1e-5,
        atol: 0.0,
        maxiter: 1000,
        m: 20,
        k: None,
        truncate: "oldest".into(),
        discard_c: false,
    };
    let mut cases = Vec::new();
    // SciPy defaults on 2-D convection-diffusion.
    for (side, peclet) in [(8, 0.5), (12, 1.5), (16, 3.0)] {
        let n = side * side;
        let mut case = base(&format!("conv2d_{side}_pe{peclet}_defaults"), n);
        case.triplets = convection_2d(side, peclet);
        cases.push(case);
    }
    // Tight tolerance, small m and k, both truncations, with CU recycling.
    for truncate in ["oldest", "smallest"] {
        for (m, k) in [(5, Some(3)), (8, Some(8)), (4, Some(1))] {
            let side = 14;
            let n = side * side;
            let mut case = base(&format!("conv2d_14_m{m}_k{k:?}_{truncate}"), n);
            case.triplets = convection_2d(side, 2.0);
            case.rtol = 1e-10;
            case.m = m;
            case.k = k;
            case.truncate = truncate.into();
            case.b2 = Some(rhs(n, 0.9));
            cases.push(case);
        }
    }
    // Matrix-free stencil, defaults and tight.
    for (n, stencil, rtol) in [
        (150, (-1.5, 3.1, -0.5), 1e-5),
        (300, (-1.2, 2.5, -0.9), 1e-10),
    ] {
        let mut case = base(&format!("stencil_{n}_rtol{rtol:e}"), n);
        case.stencil = Some(stencil);
        case.rtol = rtol;
        case.b2 = Some(rhs(n, 1.7));
        cases.push(case);
    }
    // x0, Jacobi M, atol, discard_C.
    {
        let side = 10;
        let n = side * side;
        let mut case = base("conv2d_10_x0_jacobi_atol", n);
        case.triplets = convection_2d(side, 1.0);
        case.x0 = Some((0..n).map(|i| (i as f64 * 0.11).cos()).collect());
        case.jacobi = true;
        case.rtol = 0.0;
        case.atol = 1e-9;
        case.m = 10;
        case.b2 = Some(rhs(n, 2.3));
        cases.push(case);
        let mut case = base("conv2d_10_discard_c", n);
        case.triplets = convection_2d(side, 1.0);
        case.rtol = 1e-9;
        case.m = 6;
        case.k = Some(4);
        case.discard_c = true;
        case.b2 = Some(rhs(n, 0.4));
        cases.push(case);
    }
    // Budget exhaustion: info = maxiter.
    for maxiter in [1, 3] {
        let side = 16;
        let n = side * side;
        let mut case = base(&format!("conv2d_16_maxiter{maxiter}"), n);
        case.triplets = convection_2d(side, 2.5);
        case.rtol = 1e-12;
        case.maxiter = maxiter;
        case.m = 4;
        case.k = Some(2);
        cases.push(case);
    }
    // b = 0: x = 0, info = 0.
    {
        let mut case = base("zero_rhs", 25);
        case.triplets = convection_2d(5, 1.0);
        case.b = vec![0.0; 25];
        cases.push(case);
    }
    OracleQuery { cases }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import LinearOperator, gcrotmk

q = json.load(sys.stdin)
out = []
for case in q["cases"]:
    cid = case["case_id"]
    n = int(case["n"])
    row = {"case_id": cid, "x": None, "info": None, "x2": None, "info2": None}
    try:
        if case["stencil"] is not None:
            lo, d, up = case["stencil"]
            def mv(x, lo=lo, d=d, up=up):
                x = np.asarray(x, dtype=float).reshape(-1)
                y = d * x
                y[1:] += lo * x[:-1]
                y[:-1] += up * x[1:]
                return y
            A = LinearOperator((n, n), matvec=mv, dtype=float)
            diag = np.full(n, d)
        else:
            r = [int(t[0]) for t in case["triplets"]]
            c = [int(t[1]) for t in case["triplets"]]
            v = [float(t[2]) for t in case["triplets"]]
            A = csr_array((v, (r, c)), shape=(n, n))
            diag = A.diagonal()
        M = None
        if case["jacobi"]:
            M = LinearOperator((n, n), matvec=lambda v: np.asarray(v).reshape(-1) / diag,
                               dtype=float)
        kw = dict(rtol=case["rtol"], atol=case["atol"], maxiter=case["maxiter"], m=case["m"],
                  k=case["k"], truncate=case["truncate"], discard_C=case["discard_c"])
        CU = []
        x0 = None if case["x0"] is None else np.array(case["x0"], dtype=float)
        x, info = gcrotmk(A, np.array(case["b"], dtype=float), x0=x0, M=M, CU=CU, **kw)
        row["x"] = [float(v) for v in x]
        row["info"] = int(info)
        if case["b2"] is not None:
            x2, info2 = gcrotmk(A, np.array(case["b2"], dtype=float), M=M, CU=CU, **kw)
            row["x2"] = [float(v) for v in x2]
            row["info2"] = int(info2)
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
                "failed to spawn python for gcrotmk oracle: {e}"
            );
            eprintln!("skipping gcrotmk oracle: python not available ({e})");
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
                "gcrotmk oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping gcrotmk oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "gcrotmk oracle failed: {stderr}"
        );
        eprintln!("skipping gcrotmk oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse oracle JSON"))
}

fn stencil_apply(x: &[f64], (lo, d, up): (f64, f64, f64)) -> Vec<f64> {
    let n = x.len();
    let mut y: Vec<f64> = x.iter().map(|v| d * v).collect();
    for i in 1..n {
        y[i] += lo * x[i - 1];
    }
    for i in 0..n.saturating_sub(1) {
        y[i] += up * x[i + 1];
    }
    y
}

/// fsci's two solves (the second reusing CU), or the error text.
#[allow(clippy::type_complexity)]
fn fsci_solves(case: &Case) -> Result<((Vec<f64>, usize), Option<(Vec<f64>, usize)>), String> {
    let n = case.n;
    let csr: Option<CsrMatrix> = if case.stencil.is_none() {
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for &(i, j, x) in &case.triplets {
            r.push(i);
            c.push(j);
            v.push(x);
        }
        Some(
            CooMatrix::from_triplets(Shape2D::new(n, n), v, r, c, true)
                .and_then(|coo| coo.to_csr())
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    let free = case.stencil.map(|stencil| {
        FunctionOperator::new(Shape2D::new(n, n), move |x: &[f64]| {
            Ok(stencil_apply(x, stencil))
        })
    });
    let a: &dyn LinearOperator = match (&csr, &free) {
        (Some(m), _) => m,
        (None, Some(f)) => f,
        (None, None) => unreachable!("a case is either a matrix or a stencil"),
    };
    let diag: Vec<f64> = match (&csr, case.stencil) {
        (Some(m), _) => (0..n)
            .map(|i| {
                (m.indptr()[i]..m.indptr()[i + 1])
                    .find(|&idx| m.indices()[idx] == i)
                    .map_or(0.0, |idx| m.data()[idx])
            })
            .collect(),
        (None, Some((_, d, _))) => vec![d; n],
        (None, None) => unreachable!(),
    };
    let jacobi = FunctionOperator::new(Shape2D::new(n, n), move |v: &[f64]| {
        Ok(v.iter().zip(&diag).map(|(x, d)| x / d).collect())
    });
    let preconditioner: Option<&dyn LinearOperator> = case.jacobi.then_some(&jacobi);
    let options = GcrotmkOptions {
        rtol: case.rtol,
        atol: case.atol,
        max_iter: case.maxiter,
        m: case.m,
        k: case.k,
        discard_c: case.discard_c,
        truncate: if case.truncate == "smallest" {
            GcrotmkTruncate::Smallest
        } else {
            GcrotmkTruncate::Oldest
        },
    };
    let mut cu: Vec<GcrotmkPair> = Vec::new();
    let first = gcrotmk(
        a,
        &case.b,
        case.x0.as_deref(),
        preconditioner,
        Some(&mut cu),
        options,
    )
    .map_err(|e| e.to_string())?;
    let second = match &case.b2 {
        Some(b2) => {
            let result = gcrotmk(a, b2, None, preconditioner, Some(&mut cu), options)
                .map_err(|e| e.to_string())?;
            Some((result.solution, result.info))
        }
        None => None,
    };
    Ok(((first.solution, first.info), second))
}

fn relative_inf_diff(scipy: &[f64], fsci: &[f64]) -> f64 {
    let scale = scipy.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let diff = scipy
        .iter()
        .zip(fsci)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 { diff } else { diff / scale }
}

#[test]
fn diff_sparse_gcrotmk() {
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
    let mut ledger = CompareLedger::new("diff_sparse_gcrotmk", &ARMS);
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
    let mut recycled_cases = 0;

    for case in &query.cases {
        let id = case.case_id.as_str();
        let scipy = by_id.get(id);
        let fsci = fsci_solves(case);
        if let Err(e) = &fsci {
            eprintln!("gcrotmk fsci error on {id}: {e}");
        }
        let fsci = fsci.ok();
        let first = fsci.as_ref().map(|(first, _)| first);

        if let Some((s, f)) = ledger.both(
            "info",
            id,
            scipy.and_then(|s| s.info),
            first.map(|(_, info)| *info),
        ) {
            record(
                &mut ledger,
                id,
                "info",
                if s == f { 0.0 } else { f64::INFINITY },
            );
        }
        if let Some((s, f)) = ledger.slices(
            "solution",
            id,
            scipy.and_then(|s| s.x.as_deref()),
            first.map(|(x, _)| x.as_slice()),
        ) {
            record(
                &mut ledger,
                id,
                "solution",
                relative_inf_diff(s, f) / SOLUTION_TOL,
            );
        }

        if case.b2.is_some() {
            recycled_cases += 1;
            let second = fsci.as_ref().and_then(|(_, second)| second.as_ref());
            if let Some((s, f)) = ledger.both(
                "recycled_info",
                id,
                scipy.and_then(|s| s.info2),
                second.map(|(_, info)| *info),
            ) {
                record(
                    &mut ledger,
                    id,
                    "recycled_info",
                    if s == f { 0.0 } else { f64::INFINITY },
                );
            }
            if let Some((s, f)) = ledger.slices(
                "recycled_solution",
                id,
                scipy.and_then(|s| s.x2.as_deref()),
                second.map(|(x, _)| x.as_slice()),
            ) {
                record(
                    &mut ledger,
                    id,
                    "recycled_solution",
                    relative_inf_diff(s, f) / SOLUTION_TOL,
                );
            }
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_sparse_gcrotmk".into(),
        category: "fsci_sparse::gcrotmk vs scipy.sparse.linalg.gcrotmk".into(),
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
        eprintln!("gcrotmk {} {} metric={:.3e}", d.case_id, d.arm, d.metric);
    }
    eprintln!("gcrotmk worst metric per arm (1.0 = tolerance): {max_metric:?}");
    ledger.finish(recycled_cases.min(query.cases.len()));
}
