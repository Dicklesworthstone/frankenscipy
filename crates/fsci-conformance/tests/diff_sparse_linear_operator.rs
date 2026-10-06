#![forbid(unsafe_code)]
//! Live scipy.sparse.linalg parity for `fsci_sparse::LinearOperator` (frankenscipy-6j5tz).
//!
//! Arms:
//! - `algebra`: SciPy's operator algebra (`A + B`, `A - B`, `A @ B`, `alpha * A`, `-A`,
//!   `A ** p`, `A.H`, `A.T`, `IdentityOperator`, `matmat`, `rmatmat`, nested compositions) on
//!   dense (`aslinearoperator(ndarray)`), sparse and closure-defined operands, and `matvec` /
//!   `rmatvec` of every sparse format through `aslinearoperator`, against fsci's operator types,
//!   within 1e-13 relative to the result's largest entry (a few reassociated products).
//! - `solve`: every Krylov solver on a MATRIX-FREE operator — a closure-defined stencil
//!   (SciPy's `LinearOperator(shape, matvec, rmatvec)`, fsci's `FunctionOperator`), or SciPy's
//!   `-LaplacianNd + 0.5 * IdentityOperator` against the same composition of fsci's operators.
//!   Both sides solve to `rtol = 1e-10`; the verdicts must agree, fsci's true residual must meet
//!   its contract (`≤ 2·rtol`, the factor covering a recurrence residual's drift from the true
//!   one), and the solutions must agree within what the two actual residuals certify:
//!   `x_f − x_s = A⁻¹(r_s − r_f)`, so `‖x_f − x_s‖/‖x_s‖ ≤ κ₂(A)·(‖r_s‖ + ‖r_f‖)/(‖b‖ − ‖r_s‖)`,
//!   with `κ₂` computed by SciPy on the materialized operator. (An a-priori `2κ·rtol` is wrong:
//!   SciPy's MINRES stops on `‖r‖ ≤ rtol·(‖A‖‖x‖ + ‖b‖)` and its LSQR/LSMR on a mixed
//!   `atol`/`btol` test.) The step-for-step ports land at rounding, far inside the bound.
//! - `eigen`: `eigsh` (`LA`, `SA`), `eigs` (`LM`) and `svds` on matrix-free operators, eigenvalues
//!   or singular values within 1e-9 of the largest returned magnitude (ARPACK runs to `tol = 0`).
//! - `refusal`: where SciPy raises (no `rmatvec` for `bicg`/`qmr`/`lsqr`/`lsmr`/`svds`/
//!   `rmatvec`, shape mismatches in `+`/`@`, a non-square `**`, wrong operand lengths), fsci must
//!   return an error, and where SciPy does not, fsci must not.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_sparse::{
    AdjointOperator, BsrMatrix, CooMatrix, DenseLinearOperator, DiaMatrix, DokMatrix, EigsOptions,
    EigsWhich, FormatConvertible, FunctionOperator, GcrotmkOptions, IdentityOperator,
    IterativeSolveOptions, LaplacianBoundary, LaplacianNd, LgmresOptions, LilMatrix,
    LinearOperator, PowerOperator, ProductOperator, ScaledOperator, Shape2D, SparseResult,
    SumOperator, TransposeOperator, aslinearoperator, bicg, bicgstab, bicgstab_preconditioned, cg,
    cgs, eigs, eigsh, gcrotmk, gmres, gmres_preconditioned, lgmres, lsmr, lsqr, minres, qmr, svds,
    tfqmr,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const ALGEBRA_TOL: f64 = 1.0e-13;
const SOLVE_RTOL: f64 = 1.0e-10;
const EIGEN_TOL: f64 = 1.0e-9;
const ARMS: [&str; 4] = ["algebra", "solve", "eigen", "refusal"];
const FORMATS: [&str; 7] = ["csr", "csc", "coo", "bsr", "dia", "dok", "lil"];

/// Matrix-free operators both sides build from the same description.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind")]
enum OpSpec {
    /// Tridiagonal Toeplitz `(lo, d, up)` applied by a closure; `rmatvec` optional.
    #[serde(rename = "stencil")]
    Stencil {
        n: usize,
        lo: f64,
        d: f64,
        up: f64,
        rmatvec: bool,
    },
    /// `-LaplacianNd(grid, 'dirichlet') + shift * I`, through the operator algebra.
    #[serde(rename = "neg_lap_shift")]
    NegLapShift { grid: Vec<usize>, shift: f64 },
}

impl OpSpec {
    fn n(&self) -> usize {
        match self {
            Self::Stencil { n, .. } => *n,
            Self::NegLapShift { grid, .. } => grid.iter().product(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct SolveCase {
    case_id: String,
    solver: String,
    op: OpSpec,
    b: Vec<f64>,
    jacobi: bool,
}

#[derive(Debug, Clone, Serialize)]
struct EigenCase {
    case_id: String,
    routine: String,
    which: String,
    k: usize,
    op: OpSpec,
}

#[derive(Debug, Clone, Serialize)]
struct AlgebraOperands {
    a_rows: Vec<Vec<f64>>,
    b_triplets: Vec<(usize, usize, f64)>,
    c_triplets: Vec<(usize, usize, f64)>,
    s_stencil: (f64, f64, f64),
    x4: Vec<f64>,
    x5: Vec<f64>,
    y4: Vec<f64>,
    y5: Vec<f64>,
    big_x: Vec<Vec<f64>>,
    big_y4: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    operands: AlgebraOperands,
    algebra: Vec<String>,
    refusals: Vec<String>,
    solves: Vec<SolveCase>,
    eigen: Vec<EigenCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    algebra: HashMap<String, Option<Vec<f64>>>,
    refusals: HashMap<String, Option<bool>>,
    solves: HashMap<String, SolveOutcome>,
    eigen: HashMap<String, Option<Vec<f64>>>,
}

#[derive(Debug, Clone, Deserialize)]
struct SolveOutcome {
    x: Option<Vec<f64>>,
    converged: Option<bool>,
    kappa: Option<f64>,
    /// `‖b − A·x‖ / ‖b‖` of SciPy's solution.
    residual: Option<f64>,
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

const ALGEBRA: [&str; 19] = [
    "sum_matvec",
    "difference_rmatvec",
    "product_matvec",
    "product_rmatvec",
    "scaled_matvec",
    "scaled_rmatvec",
    "negated_matvec",
    "power3_matvec",
    "power3_rmatvec",
    "power0_matvec",
    "adjoint_matvec",
    "transpose_matvec",
    "adjoint_rmatvec",
    "adjoint_of_product",
    "nested",
    "identity",
    "matmat",
    "rmatmat",
    "stencil_dot",
];

const REFUSALS: [&str; 11] = [
    "bicg_without_rmatvec",
    "qmr_without_rmatvec",
    "lsqr_without_rmatvec",
    "lsmr_without_rmatvec",
    "svds_without_rmatvec",
    "rmatvec_undefined",
    "sum_shape_mismatch",
    "product_shape_mismatch",
    "power_nonsquare",
    "matvec_wrong_length",
    "gmres_with_rmatvec_free_operator_succeeds",
];

fn generate_query() -> OracleQuery {
    let a_rows: Vec<Vec<f64>> = (0..5)
        .map(|i| {
            (0..4)
                .map(|j| ((i * 4 + j) as f64 * 0.7).sin() * 2.0 + if i == j { 3.0 } else { 0.0 })
                .collect()
        })
        .collect();
    let b_triplets = vec![
        (0, 0, 1.5),
        (0, 3, -2.0),
        (1, 1, 0.75),
        (1, 4, 3.25),
        (2, 2, -1.0),
        (2, 0, 0.5),
        (3, 4, 2.0),
        (3, 1, -0.25),
    ];
    let c_triplets = vec![
        (0, 0, 2.0),
        (1, 2, -1.5),
        (2, 1, 0.25),
        (3, 3, 4.0),
        (4, 0, -0.5),
        (4, 3, 1.25),
        (2, 3, 0.125),
    ];
    let operands = AlgebraOperands {
        a_rows,
        b_triplets,
        c_triplets,
        s_stencil: (-1.25, 3.0, 0.5),
        x4: vec![0.5, -1.0, 2.0, 0.25],
        x5: vec![1.0, -0.5, 0.75, 2.0, -1.5],
        y4: vec![-0.25, 1.5, 0.5, -2.0],
        y5: vec![0.3, 1.1, -0.7, 0.2, 0.9],
        big_x: (0..4)
            .map(|i| (0..3).map(|j| (i as f64 - j as f64) * 0.5 + 1.0).collect())
            .collect(),
        big_y4: (0..4)
            .map(|i| (0..2).map(|j| ((i + 2 * j) as f64).cos()).collect())
            .collect(),
    };
    let mut algebra: Vec<String> = ALGEBRA.iter().map(ToString::to_string).collect();
    for fmt in FORMATS {
        algebra.push(format!("format_{fmt}_matvec"));
        algebra.push(format!("format_{fmt}_rmatvec"));
    }

    let nonsym = |n: usize, rmatvec: bool| OpSpec::Stencil {
        n,
        lo: -1.5,
        d: 4.0,
        up: -0.5,
        rmatvec,
    };
    let sym = |n: usize| OpSpec::Stencil {
        n,
        lo: -1.0,
        d: 2.2,
        up: -1.0,
        rmatvec: true,
    };
    let lap = OpSpec::NegLapShift {
        grid: vec![12, 10],
        shift: 0.5,
    };
    let rhs = |n: usize| -> Vec<f64> {
        (0..n)
            .map(|i| ((i as f64) * 0.29).sin() + 0.5 + 0.01 * i as f64)
            .collect()
    };
    let mut solves = Vec::new();
    let mut solve = |solver: &str, op: OpSpec, jacobi: bool| {
        let n = op.n();
        let tag = match &op {
            OpSpec::Stencil { .. } => "stencil",
            OpSpec::NegLapShift { .. } => "neg_lap",
        };
        solves.push(SolveCase {
            case_id: format!("{solver}_{tag}{}", if jacobi { "_jacobi" } else { "" }),
            solver: solver.to_string(),
            op,
            b: rhs(n),
            jacobi,
        });
    };
    for solver in ["cg", "minres"] {
        solve(solver, sym(150), false);
        solve(solver, lap.clone(), false);
    }
    for solver in ["gmres", "lgmres", "bicgstab", "cgs", "tfqmr", "gcrotmk"] {
        solve(solver, nonsym(150, false), false);
    }
    solve("gmres", lap.clone(), false);
    for solver in ["bicg", "qmr", "lsqr", "lsmr"] {
        solve(solver, nonsym(150, true), false);
    }
    solve("gmres", nonsym(150, false), true);
    solve("bicgstab", nonsym(150, false), true);

    let eigen = vec![
        EigenCase {
            case_id: "eigsh_LA_laplacian_stencil".into(),
            routine: "eigsh".into(),
            which: "LA".into(),
            k: 4,
            op: OpSpec::Stencil {
                n: 100,
                lo: -1.0,
                d: 2.0,
                up: -1.0,
                rmatvec: true,
            },
        },
        EigenCase {
            case_id: "eigsh_SA_neg_laplacian_nd".into(),
            routine: "eigsh".into(),
            which: "SA".into(),
            k: 3,
            op: OpSpec::NegLapShift {
                grid: vec![8, 9],
                shift: 0.0,
            },
        },
        EigenCase {
            case_id: "eigs_LM_mild_nonsymmetric".into(),
            routine: "eigs".into(),
            which: "LM".into(),
            k: 3,
            op: OpSpec::Stencil {
                n: 60,
                lo: -1.1,
                d: 3.0,
                up: -0.9,
                rmatvec: true,
            },
        },
        EigenCase {
            case_id: "svds_nonsymmetric".into(),
            routine: "svds".into(),
            which: "LM".into(),
            k: 3,
            op: nonsym(80, true),
        },
    ];

    OracleQuery {
        operands,
        algebra,
        refusals: REFUSALS.iter().map(ToString::to_string).collect(),
        solves,
        eigen,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import (LaplacianNd, LinearOperator, aslinearoperator, bicg, bicgstab,
                                 cg, cgs, eigs, eigsh, gcrotmk, gmres, lgmres, lsmr, lsqr, minres,
                                 qmr, svds, tfqmr)
# IdentityOperator is public API of the LinearOperator module but not re-exported.
from scipy.sparse.linalg._interface import IdentityOperator

warnings.simplefilter("ignore")
q = json.load(sys.stdin)
ops = q["operands"]

def stencil_op(n, lo, d, up, with_rmatvec):
    def mv(x, lo=lo, d=d, up=up):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = d * x
        y[1:] += lo * x[:-1]
        y[:-1] += up * x[1:]
        return y
    def rmv(x, lo=lo, d=d, up=up):
        return mv(x, up, d, lo)
    if with_rmatvec:
        return LinearOperator((n, n), matvec=mv, rmatvec=rmv, dtype=float)
    return LinearOperator((n, n), matvec=mv, dtype=float)

def build(spec):
    if spec["kind"] == "stencil":
        return stencil_op(spec["n"], spec["lo"], spec["d"], spec["up"], spec["rmatvec"])
    L = LaplacianNd(tuple(spec["grid"]), boundary_conditions="dirichlet", dtype=np.float64)
    n = int(np.prod(spec["grid"]))
    return -L + spec["shift"] * IdentityOperator((n, n), dtype=np.float64)

def coo(trips, shape):
    r = [t[0] for t in trips]; c = [t[1] for t in trips]; v = [t[2] for t in trips]
    return sp.coo_array((v, (r, c)), shape=shape)

A = aslinearoperator(np.array(ops["a_rows"], dtype=float))
B = aslinearoperator(coo(ops["b_triplets"], (4, 5)).tocsr())
C = aslinearoperator(coo(ops["c_triplets"], (5, 4)))
lo, d, up = ops["s_stencil"]
S = stencil_op(5, lo, d, up, True)
x4 = np.array(ops["x4"]); x5 = np.array(ops["x5"]); y4 = np.array(ops["y4"]); y5 = np.array(ops["y5"])
X = np.array(ops["big_x"]); Y4 = np.array(ops["big_y4"])
exprs = {
    "sum_matvec": lambda: (A + C).matvec(x4),
    "difference_rmatvec": lambda: (A - C).rmatvec(y5),
    "product_matvec": lambda: (A @ B).matvec(x5),
    "product_rmatvec": lambda: (A @ B).rmatvec(y5),
    "scaled_matvec": lambda: (2.5 * S).matvec(x5),
    "scaled_rmatvec": lambda: (2.5 * S).rmatvec(y5),
    "negated_matvec": lambda: (-S).matvec(x5),
    "power3_matvec": lambda: (S ** 3).matvec(x5),
    "power3_rmatvec": lambda: (S ** 3).rmatvec(y5),
    "power0_matvec": lambda: (S ** 0).matvec(x5),
    "adjoint_matvec": lambda: A.H.matvec(y5),
    "transpose_matvec": lambda: A.T.matvec(y5),
    "adjoint_rmatvec": lambda: A.H.rmatvec(x4),
    "adjoint_of_product": lambda: (B.H @ A.H).matvec(y5),
    "nested": lambda: ((A @ B) * 2.0 + S ** 2).H.matvec(y5),
    "identity": lambda: IdentityOperator((5, 5)).matvec(x5),
    "matmat": lambda: A.matmat(X).ravel(),
    "rmatmat": lambda: B.rmatmat(Y4).ravel(),
    "stencil_dot": lambda: S.dot(x5),
}
Cs = coo(ops["c_triplets"], (5, 4))
for fmt in ["csr", "csc", "coo", "bsr", "dia", "dok", "lil"]:
    M = aslinearoperator(Cs.asformat(fmt))
    exprs[f"format_{fmt}_matvec"] = (lambda M=M: M.matvec(x4))
    exprs[f"format_{fmt}_rmatvec"] = (lambda M=M: M.rmatvec(y5))
algebra = {}
for name in q["algebra"]:
    try:
        algebra[name] = [float(v) for v in np.asarray(exprs[name]()).ravel()]
    except Exception as e:
        sys.stderr.write(f"oracle algebra {name}: {e!r}\n")
        algebra[name] = None

free = stencil_op(20, -1.5, 4.0, -0.5, False)
bvec = np.ones(20)
refusal_calls = {
    "bicg_without_rmatvec": lambda: bicg(free, bvec, rtol=1e-10),
    "qmr_without_rmatvec": lambda: qmr(free, bvec, rtol=1e-10),
    "lsqr_without_rmatvec": lambda: lsqr(free, bvec),
    "lsmr_without_rmatvec": lambda: lsmr(free, bvec),
    "svds_without_rmatvec": lambda: svds(free, k=2),
    "rmatvec_undefined": lambda: free.rmatvec(bvec),
    "sum_shape_mismatch": lambda: A + B,
    "product_shape_mismatch": lambda: A @ C,
    "power_nonsquare": lambda: A ** 2,
    "matvec_wrong_length": lambda: A.matvec(x5),
    "gmres_with_rmatvec_free_operator_succeeds": lambda: gmres(free, bvec, rtol=1e-10),
}
refusals = {}
for name in q["refusals"]:
    try:
        refusal_calls[name]()
        refusals[name] = False
    except Exception:
        refusals[name] = True

solves = {}
for case in q["solves"]:
    cid = case["case_id"]
    out = {"x": None, "converged": None, "kappa": None, "residual": None}
    try:
        op = build(case["op"])
        n = op.shape[0]
        b = np.array(case["b"], dtype=float)
        dense = op.matmat(np.eye(n))
        out["kappa"] = float(np.linalg.cond(dense))
        M = None
        if case["jacobi"]:
            diag = np.diag(dense).copy()
            M = LinearOperator((n, n), matvec=lambda v, diag=diag: np.asarray(v).reshape(-1) / diag,
                               dtype=float)
        s = case["solver"]
        rt = 1e-10
        if s == "cg":
            x, info = cg(op, b, rtol=rt, atol=0.0, maxiter=10 * n)
        elif s == "minres":
            x, info = minres(op, b, rtol=rt, maxiter=10 * n)
        elif s == "gmres":
            x, info = gmres(op, b, rtol=rt, atol=0.0, M=M, maxiter=10 * n)
        elif s == "lgmres":
            x, info = lgmres(op, b, rtol=rt, atol=0.0)
        elif s == "bicgstab":
            x, info = bicgstab(op, b, rtol=rt, atol=0.0, M=M, maxiter=10 * n)
        elif s == "cgs":
            x, info = cgs(op, b, rtol=rt, atol=0.0, maxiter=10 * n)
        elif s == "tfqmr":
            x, info = tfqmr(op, b, rtol=rt, atol=0.0)
        elif s == "gcrotmk":
            x, info = gcrotmk(op, b, rtol=rt, atol=0.0)
        elif s == "bicg":
            x, info = bicg(op, b, rtol=rt, atol=0.0, maxiter=10 * n)
        elif s == "qmr":
            x, info = qmr(op, b, rtol=rt, atol=0.0, maxiter=10 * n)
        elif s == "lsqr":
            res = lsqr(op, b, atol=rt, btol=rt, iter_lim=10 * n)
            x, info = res[0], (0 if res[1] in (1, 2) else 1)
        elif s == "lsmr":
            res = lsmr(op, b, atol=rt, btol=rt, maxiter=10 * n)
            x, info = res[0], (0 if res[1] in (1, 2) else 1)
        else:
            raise ValueError(s)
        out["x"] = [float(v) for v in x]
        out["converged"] = bool(info == 0)
        out["residual"] = float(np.linalg.norm(b - op.matvec(np.asarray(x))) / np.linalg.norm(b))
    except Exception as e:
        sys.stderr.write(f"oracle solve {cid}: {e!r}\n")
    solves[cid] = out

eigen = {}
for case in q["eigen"]:
    cid = case["case_id"]
    try:
        op = build(case["op"])
        k = case["k"]
        if case["routine"] == "eigsh":
            vals = np.sort(eigsh(op, k=k, which=case["which"], return_eigenvectors=False))
        elif case["routine"] == "eigs":
            vals = eigs(op, k=k, which=case["which"], return_eigenvectors=False)
            vals = np.sort(vals.real)
        else:
            vals = np.sort(svds(op, k=k, return_singular_vectors=False))
        eigen[cid] = [float(v) for v in vals]
    except Exception as e:
        sys.stderr.write(f"oracle eigen {cid}: {e!r}\n")
        eigen[cid] = None

print(json.dumps({"algebra": algebra, "refusals": refusals, "solves": solves, "eigen": eigen}))
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
                "failed to spawn python for LinearOperator oracle: {e}"
            );
            eprintln!("skipping LinearOperator oracle: python not available ({e})");
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
                "LinearOperator oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping LinearOperator oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "LinearOperator oracle failed: {stderr}"
        );
        eprintln!("skipping LinearOperator oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse oracle JSON"))
}

fn stencil_apply(x: &[f64], lo: f64, d: f64, up: f64) -> Vec<f64> {
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

fn stencil_operator(
    n: usize,
    (lo, d, up): (f64, f64, f64),
    with_rmatvec: bool,
) -> FunctionOperator<'static> {
    let op = FunctionOperator::new(Shape2D::new(n, n), move |x: &[f64]| {
        Ok(stencil_apply(x, lo, d, up))
    });
    if with_rmatvec {
        op.with_rmatvec(move |x: &[f64]| Ok(stencil_apply(x, up, d, lo)))
    } else {
        op
    }
}

fn build(spec: &OpSpec) -> Box<dyn LinearOperator> {
    match spec {
        OpSpec::Stencil {
            n,
            lo,
            d,
            up,
            rmatvec,
        } => Box::new(stencil_operator(*n, (*lo, *d, *up), *rmatvec)),
        OpSpec::NegLapShift { grid, shift } => {
            let lap = LaplacianNd::new(grid, LaplacianBoundary::Dirichlet).expect("LaplacianNd");
            let n: usize = grid.iter().product();
            Box::new(
                SumOperator::new(
                    ScaledOperator::new(lap, -1.0),
                    ScaledOperator::new(IdentityOperator::new(n), *shift),
                )
                .expect("same shape"),
            )
        }
    }
}

fn coo(triplets: &[(usize, usize, f64)], rows: usize, cols: usize) -> CooMatrix {
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for &(i, j, x) in triplets {
        r.push(i);
        c.push(j);
        v.push(x);
    }
    CooMatrix::from_triplets(Shape2D::new(rows, cols), v, r, c, false).expect("coo")
}

fn flatten(rows: &[Vec<f64>]) -> Vec<f64> {
    rows.iter().flatten().copied().collect()
}

/// fsci's value of one algebra expression.
#[allow(clippy::too_many_lines)]
fn fsci_algebra(name: &str, ops: &AlgebraOperands) -> SparseResult<Vec<f64>> {
    let a = DenseLinearOperator::from_rows(&ops.a_rows)?;
    let b = coo(&ops.b_triplets, 4, 5).to_csr()?;
    let c = coo(&ops.c_triplets, 5, 4);
    let s = stencil_operator(5, ops.s_stencil, true);
    let (x4, x5, y5) = (&ops.x4, &ops.x5, &ops.y5);
    if let Some(rest) = name.strip_prefix("format_") {
        let (fmt, product) = rest.split_once('_').expect("format_<fmt>_<product>");
        let csr = c.to_csr()?;
        // SciPy's `aslinearoperator(C.asformat(fmt))`, format by format.
        let op: Box<dyn LinearOperator> = match fmt {
            "csr" => aslinearoperator(csr),
            "csc" => aslinearoperator(c.to_csc()?),
            "coo" => aslinearoperator(c.clone()),
            "bsr" => aslinearoperator(BsrMatrix::from_triplets(
                c.shape(),
                Shape2D::new(1, 1),
                c.data().to_vec(),
                c.row_indices().to_vec(),
                c.col_indices().to_vec(),
            )?),
            "dia" => aslinearoperator(DiaMatrix::from_triplets(
                c.shape(),
                c.data().to_vec(),
                c.row_indices().to_vec(),
                c.col_indices().to_vec(),
            )?),
            "dok" => aslinearoperator(DokMatrix::from_triplets(
                c.shape(),
                c.data().to_vec(),
                c.row_indices().to_vec(),
                c.col_indices().to_vec(),
            )?),
            "lil" => aslinearoperator(LilMatrix::from_triplets(
                c.shape(),
                c.data().to_vec(),
                c.row_indices().to_vec(),
                c.col_indices().to_vec(),
            )?),
            other => panic!("unknown format {other}"),
        };
        return if product == "matvec" {
            op.matvec(x4)
        } else {
            op.rmatvec(y5)
        };
    }
    match name {
        "sum_matvec" => SumOperator::new(&a, &c)?.matvec(x4),
        "difference_rmatvec" => SumOperator::new(&a, ScaledOperator::new(&c, -1.0))?.rmatvec(y5),
        "product_matvec" => ProductOperator::new(&a, &b)?.matvec(x5),
        "product_rmatvec" => ProductOperator::new(&a, &b)?.rmatvec(y5),
        "scaled_matvec" => ScaledOperator::new(&s, 2.5).matvec(x5),
        "scaled_rmatvec" => ScaledOperator::new(&s, 2.5).rmatvec(y5),
        "negated_matvec" => ScaledOperator::new(&s, -1.0).matvec(x5),
        "power3_matvec" => PowerOperator::new(&s, 3)?.matvec(x5),
        "power3_rmatvec" => PowerOperator::new(&s, 3)?.rmatvec(y5),
        "power0_matvec" => PowerOperator::new(&s, 0)?.matvec(x5),
        "adjoint_matvec" => AdjointOperator::new(&a).matvec(y5),
        "transpose_matvec" => TransposeOperator::new(&a).matvec(y5),
        "adjoint_rmatvec" => AdjointOperator::new(&a).rmatvec(x4),
        "adjoint_of_product" => {
            ProductOperator::new(AdjointOperator::new(&b), AdjointOperator::new(&a))?.matvec(y5)
        }
        "nested" => AdjointOperator::new(SumOperator::new(
            ScaledOperator::new(ProductOperator::new(&a, &b)?, 2.0),
            PowerOperator::new(&s, 2)?,
        )?)
        .matvec(y5),
        "identity" => IdentityOperator::new(5).matvec(x5),
        "matmat" => a.matmat(&ops.big_x).map(|rows| flatten(&rows)),
        "rmatmat" => b.rmatmat(&ops.big_y4).map(|rows| flatten(&rows)),
        "stencil_dot" => s.matvec(x5),
        other => panic!("unknown algebra expression {other}"),
    }
}

/// Whether fsci refuses the call SciPy is asked about.
fn fsci_refuses(name: &str, ops: &AlgebraOperands) -> bool {
    let free = stencil_operator(20, (-1.5, 4.0, -0.5), false);
    let b = vec![1.0; 20];
    let a = DenseLinearOperator::from_rows(&ops.a_rows).expect("dense");
    let bm = coo(&ops.b_triplets, 4, 5).to_csr().expect("csr");
    let c = coo(&ops.c_triplets, 5, 4);
    let options = IterativeSolveOptions {
        tol: 1e-10,
        ..IterativeSolveOptions::default()
    };
    match name {
        "bicg_without_rmatvec" => bicg(&free, &b, None, options).is_err(),
        "qmr_without_rmatvec" => qmr(&free, &b, None, options).is_err(),
        "lsqr_without_rmatvec" => lsqr(&free, &b, options).is_err(),
        "lsmr_without_rmatvec" => lsmr(&free, &b, options).is_err(),
        "svds_without_rmatvec" => svds(&free, 2, EigsOptions::default()).is_err(),
        "rmatvec_undefined" => free.rmatvec(&b).is_err(),
        "sum_shape_mismatch" => SumOperator::new(&a, &bm).is_err(),
        "product_shape_mismatch" => ProductOperator::new(&a, &c).is_err(),
        "power_nonsquare" => PowerOperator::new(&a, 2).is_err(),
        "matvec_wrong_length" => a.matvec(&ops.x5).is_err(),
        "gmres_with_rmatvec_free_operator_succeeds" => gmres(&free, &b, None, options).is_err(),
        other => panic!("unknown refusal case {other}"),
    }
}

/// fsci's solution and convergence verdict for one solve case.
fn fsci_solve(case: &SolveCase) -> SparseResult<(Vec<f64>, bool)> {
    let op = build(&case.op);
    let op = op.as_ref();
    let n = case.op.n();
    let b = &case.b;
    let options = IterativeSolveOptions {
        tol: SOLVE_RTOL,
        max_iter: Some(10 * n),
        ..IterativeSolveOptions::default()
    };
    let diag: Vec<f64> = {
        let dense = op.to_dense()?;
        (0..n).map(|i| dense[i][i]).collect()
    };
    let jacobi = |r: &[f64]| -> SparseResult<Vec<f64>> {
        Ok(r.iter().zip(&diag).map(|(v, d)| v / d).collect())
    };
    let result = match case.solver.as_str() {
        "cg" => cg(op, b, None, options)?,
        "minres" => minres(op, b, None, options)?,
        "gmres" if case.jacobi => gmres_preconditioned(op, b, jacobi, None, None, options)?,
        "gmres" => gmres(op, b, None, options)?,
        "lgmres" => lgmres(
            op,
            b,
            None,
            LgmresOptions {
                tol: SOLVE_RTOL,
                ..LgmresOptions::default()
            },
        )?,
        "bicgstab" if case.jacobi => bicgstab_preconditioned(op, b, jacobi, None, options)?,
        "bicgstab" => bicgstab(op, b, None, options)?,
        "cgs" => cgs(op, b, None, options)?,
        "tfqmr" => tfqmr(op, b, None, options)?,
        "gcrotmk" => {
            let result = gcrotmk(
                op,
                b,
                None,
                None,
                None,
                GcrotmkOptions {
                    rtol: SOLVE_RTOL,
                    ..GcrotmkOptions::default()
                },
            )?;
            return Ok((result.solution, result.converged));
        }
        "bicg" => bicg(op, b, None, options)?,
        "qmr" => qmr(op, b, None, options)?,
        "lsqr" => lsqr(op, b, options)?,
        "lsmr" => lsmr(op, b, options)?,
        other => panic!("unknown solver {other}"),
    };
    Ok((result.solution, result.converged))
}

fn fsci_eigen(case: &EigenCase) -> SparseResult<Vec<f64>> {
    let op = build(&case.op);
    let which: EigsWhich = case.which.parse()?;
    let options = EigsOptions {
        which,
        ..EigsOptions::default()
    };
    let mut values = match case.routine.as_str() {
        "eigsh" => eigsh(op.as_ref(), case.k, options)?.eigenvalues,
        "eigs" => eigs(op.as_ref(), case.k, options)?.eigenvalues,
        _ => svds(op.as_ref(), case.k, EigsOptions::default())?.singular_values,
    };
    values.sort_by(f64::total_cmp);
    Ok(values)
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
#[allow(clippy::too_many_lines)]
fn diff_sparse_linear_operator() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let start = Instant::now();
    let mut ledger = CompareLedger::new("diff_sparse_linear_operator", &ARMS);
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

    for name in &query.algebra {
        let scipy = oracle.algebra.get(name).cloned().flatten();
        let fsci = fsci_algebra(name, &query.operands);
        if let Err(e) = &fsci {
            eprintln!("LinearOperator algebra {name}: fsci error {e}");
        }
        let fsci = fsci.ok();
        if let Some((s, f)) = ledger.slices("algebra", name, scipy.as_deref(), fsci.as_deref()) {
            let scale = s.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
            let diff = s
                .iter()
                .zip(f)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0_f64, f64::max);
            record(&mut ledger, name, "algebra", diff / scale / ALGEBRA_TOL);
        }
    }

    for name in &query.refusals {
        match oracle.refusals.get(name).copied().flatten() {
            Some(scipy_raised) => {
                let fsci_refused = fsci_refuses(name, &query.operands);
                record(
                    &mut ledger,
                    name,
                    "refusal",
                    if scipy_raised == fsci_refused {
                        0.0
                    } else {
                        f64::INFINITY
                    },
                );
            }
            None => ledger.oracle_missing("refusal", name, "no verdict"),
        }
    }

    for case in &query.solves {
        let id = case.case_id.as_str();
        let scipy = oracle.solves.get(id);
        let fsci = fsci_solve(case);
        if let Err(e) = &fsci {
            eprintln!("LinearOperator solve {id}: fsci error {e}");
        }
        let scipy_value =
            scipy.and_then(|s| Some((s.x.as_ref()?, s.converged?, s.kappa?, s.residual?)));
        if let Some(((sx, s_conv, kappa, s_residual), (fx, f_conv))) =
            ledger.both("solve", id, scipy_value, fsci.ok())
        {
            // x_f − x_s = A⁻¹(r_s − r_f), and ‖x_s‖ ≥ (‖b‖ − ‖r_s‖)/‖A‖, so the two solutions
            // can differ by at most κ·(‖r_s‖ + ‖r_f‖)/(‖b‖ − ‖r_s‖), relative to ‖x_s‖.
            let f_residual = {
                let op = build(&case.op);
                let ax = op.matvec(&fx).expect("matvec");
                let r: Vec<f64> = case.b.iter().zip(&ax).map(|(p, q)| p - q).collect();
                norm2(&r) / norm2(&case.b)
            };
            let bound =
                kappa * (s_residual + f_residual) / (1.0 - s_residual) + 8.0 * f64::EPSILON * kappa;
            eprintln!(
                "LinearOperator solve {id}: residuals fsci {f_residual:.3e} SciPy {s_residual:.3e}"
            );
            let diff: Vec<f64> = sx.iter().zip(&fx).map(|(p, q)| p - q).collect();
            let relative = norm2(&diff) / norm2(sx);
            let metric = if s_conv && f_conv && sx.len() == fx.len() {
                // fsci's own contract too: a converged solve has ‖b − A·x‖ ≤ rtol·‖b‖ (twice
                // that, for the drift of a recurrence-updated residual from the true one).
                (relative / bound).max(f_residual / (2.0 * SOLVE_RTOL))
            } else {
                eprintln!("LinearOperator solve {id}: converged fsci {f_conv} SciPy {s_conv}");
                f64::INFINITY
            };
            eprintln!(
                "LinearOperator solve {id}: rel diff {relative:.3e} (bound {bound:.3e}, kappa {kappa:.2})"
            );
            record(&mut ledger, id, "solve", metric);
        }
    }

    for case in &query.eigen {
        let id = case.case_id.as_str();
        let scipy = oracle.eigen.get(id).cloned().flatten();
        let fsci = fsci_eigen(case);
        if let Err(e) = &fsci {
            eprintln!("LinearOperator eigen {id}: fsci error {e}");
        }
        let fsci = fsci.ok();
        if let Some((s, f)) = ledger.slices("eigen", id, scipy.as_deref(), fsci.as_deref()) {
            let scale = s
                .iter()
                .fold(0.0_f64, |m, v| m.max(v.abs()))
                .max(f64::MIN_POSITIVE);
            let diff = s
                .iter()
                .zip(f)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0_f64, f64::max);
            record(&mut ledger, id, "eigen", diff / scale / EIGEN_TOL);
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let case_count =
        query.algebra.len() + query.refusals.len() + query.solves.len() + query.eigen.len();
    let log = DiffLog {
        test_id: "diff_sparse_linear_operator".into(),
        category: "fsci_sparse::LinearOperator (algebra, matrix-free solvers and eigensolvers) \
                   vs scipy.sparse.linalg.LinearOperator"
            .into(),
        case_count,
        compared: ledger.counts().clone(),
        max_metric: max_metric.clone(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);
    for d in diffs.iter().filter(|d| !d.pass) {
        eprintln!(
            "LinearOperator mismatch: {} {} metric={}",
            d.case_id, d.arm, d.metric
        );
    }
    eprintln!("LinearOperator worst metric per arm (1.0 = tolerance): {max_metric:?}");
    ledger.finish(query.eigen.len());
}
