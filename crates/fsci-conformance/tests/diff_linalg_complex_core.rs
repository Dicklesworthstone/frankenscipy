#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `fsci_linalg::complex`, the complex128 dense core
//! (frankenscipy-1ksfv.17): solve, lu, lu_factor/lu_solve (trans 0/1/2), det, inv, cholesky,
//! qr (full and economic), hessenberg, schur(output='complex'), eig, eigvals, eigh, eigvalsh,
//! svd, lstsq, pinv, norm, expm, sqrtm, logm, and the REAL-input logm/sqrtm whose results SciPy
//! returns as complex arrays when the matrix has a negative real eigenvalue.
//!
//! The LAPACK-transcribed kernels (zgetf2, zgeqr2/zung2r, zgebal, zgehd2/zunghr, zlahqr,
//! ztrevc3 + zgeev normalization) are compared ENTRY BY ENTRY: same pivots, same `R` signs, same
//! eigenvalue order, same eigenvector phase. Eigen- and singular vectors of `eigh` and `svd`
//! are unique only up to a unit-modulus factor (LAPACK does not normalize their phase), so those
//! are compared through `|uᴴ·u'| = 1` per vector and, for `svd`, through the phase agreement of
//! each left/right pair and the projector onto the completed (full_matrices) columns.
//!
//! Every case must be compared: a SciPy failure or an fsci error is a FAILED case.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_linalg::complex::{self as cx, C64, ComplexMatrix, MaybeComplexMatrix};
use fsci_linalg::{DecompOptions, MatrixAssumption, NormKind, SolveOptions};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-002";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

/// Entry-wise agreement for direct factors and solutions, relative to max(1, max|SciPy|).
const DIRECT_TOL: f64 = 1.0e-12;
/// Eigenvectors, Schur vectors and matrix functions, whose error scales with the conditioning
/// of the eigenproblem (the corpus is random and well separated, so this is still tight).
const SPECTRAL_TOL: f64 = 1.0e-10;

const ARMS: &[&str] = &[
    "solve",
    "solve_pos",
    "lu",
    "lu_solve",
    "det",
    "inv",
    "cholesky",
    "qr",
    "qr_economic",
    "hessenberg",
    "schur",
    "eig",
    "eigvals",
    "eigh",
    "svd",
    "lstsq",
    "pinv",
    "norm",
    "expm",
    "sqrtm",
    "logm",
    "logm_real",
    "sqrtm_real",
];

type Pair = [f64; 2];

#[derive(Debug, Clone, Serialize)]
struct Query {
    id: String,
    op: String,
    a: Vec<Vec<Pair>>,
    b: Option<Vec<Pair>>,
    real: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleArm {
    id: String,
    ok: bool,
    err: Option<String>,
    #[serde(default)]
    m: BTreeMap<String, Vec<Vec<Pair>>>,
    #[serde(default)]
    v: BTreeMap<String, Vec<Pair>>,
    #[serde(default)]
    s: BTreeMap<String, f64>,
    cplx: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    id: String,
    op: String,
    max_err: f64,
    pass: bool,
    reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    max_err_by_op: BTreeMap<String, f64>,
    pass: bool,
    timestamp_ms: u128,
    cases: Vec<CaseDiff>,
}

// ── corpus ────────────────────────────────────────────────────────────────

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn random_complex(rows: usize, cols: usize, seed: u64) -> ComplexMatrix {
    let mut rng = Lcg(seed);
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| C64::new(rng.next(), rng.next()))
                .collect()
        })
        .collect()
}

fn random_vec(n: usize, seed: u64) -> Vec<C64> {
    let mut rng = Lcg(seed);
    (0..n).map(|_| C64::new(rng.next(), rng.next())).collect()
}

fn hpd(n: usize, seed: u64) -> ComplexMatrix {
    let b = random_complex(n, n, seed);
    let mut h = cx::matmul(&cx::conj_transpose(&b), &b).expect("BᴴB");
    for (i, row) in h.iter_mut().enumerate() {
        row[i] += C64::new(n as f64, 0.0);
    }
    h
}

fn hermitian(n: usize, seed: u64) -> ComplexMatrix {
    let b = random_complex(n, n, seed);
    let bh = cx::conj_transpose(&b);
    (0..n)
        .map(|i| (0..n).map(|j| b[i][j] + bh[i][j]).collect())
        .collect()
}

fn scale_matrix(a: &ComplexMatrix, f: f64) -> ComplexMatrix {
    a.iter()
        .map(|r| r.iter().map(|z| z * f).collect())
        .collect()
}

/// A matrix whose rows/columns 1 and 3 isolate eigenvalues, so `zgebal` permutes.
fn isolatable() -> ComplexMatrix {
    let z = C64::new(0.0, 0.0);
    let c = |re: f64, im: f64| C64::new(re, im);
    vec![
        vec![
            c(1.0, 0.5),
            c(2.0, 0.0),
            c(0.3, -1.0),
            c(1.0, 1.0),
            c(0.5, 0.0),
        ],
        vec![z, c(-2.0, 1.0), z, z, z],
        vec![
            c(0.7, 0.0),
            c(1.0, -1.0),
            c(3.0, 0.0),
            c(2.0, 0.0),
            c(-1.0, 0.2),
        ],
        vec![z, c(0.5, 0.5), z, c(4.0, -2.0), z],
        vec![
            c(-1.0, 0.0),
            c(0.0, 2.0),
            c(0.5, 0.5),
            c(1.0, 0.0),
            c(0.2, 0.0),
        ],
    ]
}

/// Entries spanning twelve orders of magnitude, so `zgebal` scales.
fn badly_scaled(n: usize, seed: u64) -> ComplexMatrix {
    let base = random_complex(n, n, seed);
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| base[i][j] * 10f64.powi(3 * (i as i32 - j as i32)))
                .collect()
        })
        .collect()
}

fn upper_triangular(n: usize, seed: u64) -> ComplexMatrix {
    let mut a = random_complex(n, n, seed);
    for (i, row) in a.iter_mut().enumerate() {
        for v in row.iter_mut().take(i) {
            *v = C64::new(0.0, 0.0);
        }
        row[i] += C64::new(2.0, 0.0);
    }
    a
}

struct Case {
    id: String,
    op: &'static str,
    a: ComplexMatrix,
    b: Option<Vec<C64>>,
    real: Option<Vec<Vec<f64>>>,
}

fn case(id: String, op: &'static str, a: ComplexMatrix, b: Option<Vec<C64>>) -> Case {
    Case {
        id,
        op,
        a,
        b,
        real: None,
    }
}

fn real_case(id: &str, op: &'static str, m: Vec<Vec<f64>>) -> Case {
    Case {
        id: id.to_string(),
        op,
        a: cx::to_complex(&m),
        b: None,
        real: Some(m),
    }
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for (k, &n) in [1usize, 2, 3, 4, 5, 8, 12, 20].iter().enumerate() {
        let seed = 1000 + k as u64;
        let a = random_complex(n, n, seed);
        let b = random_vec(n, seed + 77);
        for op in [
            "solve",
            "lu",
            "lu_solve",
            "det",
            "inv",
            "qr",
            "qr_economic",
            "hessenberg",
            "schur",
            "eig",
            "eigvals",
            "svd",
            "pinv",
            "norm",
            "expm",
            "sqrtm",
            "logm",
        ] {
            let rhs = matches!(op, "solve" | "lu_solve").then(|| b.clone());
            out.push(case(format!("gen{n}_{op}"), op, a.clone(), rhs));
        }
        out.push(case(
            format!("gen{n}_lstsq"),
            "lstsq",
            a.clone(),
            Some(b.clone()),
        ));
    }
    for (k, &(m, n)) in [(6usize, 3usize), (3, 6), (10, 4)].iter().enumerate() {
        let seed = 2000 + k as u64;
        let a = random_complex(m, n, seed);
        for op in ["lu", "qr", "qr_economic", "svd", "pinv", "norm"] {
            out.push(case(format!("rect{m}x{n}_{op}"), op, a.clone(), None));
        }
        out.push(case(
            format!("rect{m}x{n}_lstsq"),
            "lstsq",
            a.clone(),
            Some(random_vec(m, seed + 5)),
        ));
    }
    for (k, &n) in [2usize, 4, 7].iter().enumerate() {
        let h = hpd(n, 3000 + k as u64);
        out.push(case(
            format!("hpd{n}_cholesky"),
            "cholesky",
            h.clone(),
            None,
        ));
        out.push(case(
            format!("hpd{n}_solve_pos"),
            "solve_pos",
            h.clone(),
            Some(random_vec(n, 3100 + k as u64)),
        ));
        out.push(case(format!("hpd{n}_eigh"), "eigh", h, None));
    }
    out.push(case("herm6_eigh".into(), "eigh", hermitian(6, 3300), None));
    let big = scale_matrix(&random_complex(6, 6, 3400), 12.0);
    out.push(case("expm_scaled".into(), "expm", big, None));
    let iso = isolatable();
    for op in ["eig", "eigvals", "schur"] {
        out.push(case(format!("isolatable_{op}"), op, iso.clone(), None));
    }
    let scaled = badly_scaled(5, 3500);
    for op in ["eig", "eigvals"] {
        out.push(case(format!("badly_scaled_{op}"), op, scaled.clone(), None));
    }
    let tri = upper_triangular(5, 3600);
    for op in ["eig", "sqrtm", "logm"] {
        out.push(case(format!("triu5_{op}"), op, tri.clone(), None));
    }
    // Real input whose logarithm / square root SciPy returns as complex (or real).
    let reals: Vec<(&str, Vec<Vec<f64>>)> = vec![
        ("diag_neg", vec![vec![-1.0, 0.0], vec![0.0, 1.0]]),
        ("diag_neg4", vec![vec![-4.0, 0.0], vec![0.0, 1.0]]),
        ("triu_neg", vec![vec![4.0, 1.0], vec![0.0, -4.0]]),
        ("mixed", vec![vec![1.0, 2.0], vec![3.0, -4.0]]),
        (
            "rotation",
            vec![
                vec![0.6_f64.cos(), -0.6_f64.sin()],
                vec![0.6_f64.sin(), 0.6_f64.cos()],
            ],
        ),
        (
            "spd3",
            vec![
                vec![4.0, 1.0, 0.5],
                vec![1.0, 3.0, 0.2],
                vec![0.5, 0.2, 2.0],
            ],
        ),
        (
            "general4",
            vec![
                vec![0.3, -1.2, 2.0, 0.1],
                vec![1.5, -0.4, 0.3, -2.0],
                vec![-0.7, 0.9, -3.0, 0.5],
                vec![0.2, 1.1, 0.4, 1.7],
            ],
        ),
    ];
    for (name, m) in reals {
        out.push(real_case(
            &format!("real_{name}_logm"),
            "logm_real",
            m.clone(),
        ));
        out.push(real_case(&format!("real_{name}_sqrtm"), "sqrtm_real", m));
    }
    out
}

// ── oracle ────────────────────────────────────────────────────────────────

const ORACLE: &str = r#"
import json, sys, warnings
import numpy as np
import scipy.linalg as sl

def M(x): return np.array([[complex(p[0], p[1]) for p in row] for row in x], dtype=complex).reshape(len(x), -1 if len(x) else 0)
def V(x): return np.array([complex(p[0], p[1]) for p in x], dtype=complex)
def m(x): return [[[float(z.real), float(z.imag)] for z in row] for row in np.atleast_2d(np.asarray(x, dtype=complex))]
def v(x): return [[float(z.real), float(z.imag)] for z in np.asarray(x, dtype=complex).ravel()]

out = []
for q in json.load(sys.stdin):
    arm = {"id": q["id"], "ok": False, "err": None, "m": {}, "v": {}, "s": {}, "cplx": None}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            op = q["op"]
            a = M(q["a"])
            b = V(q["b"]) if q["b"] is not None else None
            if op == "solve":
                arm["v"]["x"] = v(sl.solve(a, b))
            elif op == "solve_pos":
                arm["v"]["x"] = v(sl.solve(a, b, assume_a="pos"))
            elif op == "lu":
                p, l, u = sl.lu(a)
                arm["m"].update(p=m(p), l=m(l), u=m(u))
            elif op == "lu_solve":
                lu, piv = sl.lu_factor(a)
                arm["m"]["lu"] = m(lu)
                arm["v"]["piv"] = v(piv.astype(float))
                for t in (0, 1, 2):
                    arm["v"][f"x{t}"] = v(sl.lu_solve((lu, piv), b, trans=t))
            elif op == "det":
                arm["v"]["det"] = v([sl.det(a)])
            elif op == "inv":
                arm["m"]["inv"] = m(sl.inv(a))
            elif op == "cholesky":
                arm["m"]["u"] = m(sl.cholesky(a, lower=False))
                arm["m"]["l"] = m(sl.cholesky(a, lower=True))
            elif op == "qr":
                q_, r_ = sl.qr(a)
                arm["m"].update(q=m(q_), r=m(r_))
            elif op == "qr_economic":
                q_, r_ = sl.qr(a, mode="economic")
                arm["m"].update(q=m(q_), r=m(r_))
            elif op == "hessenberg":
                h, qq = sl.hessenberg(a, calc_q=True)
                arm["m"].update(h=m(h), q=m(qq))
            elif op == "schur":
                t, z = sl.schur(a, output="complex")
                arm["m"].update(t=m(t), z=m(z))
            elif op == "eig":
                w, vr = sl.eig(a)
                arm["v"]["w"] = v(w)
                arm["m"]["v"] = m(vr)
            elif op == "eigvals":
                arm["v"]["w"] = v(sl.eigvals(a))
            elif op == "eigh":
                w, vv = sl.eigh(a)
                arm["v"]["w"] = v(w)
                arm["m"]["v"] = m(vv)
                arm["v"]["wh"] = v(sl.eigvalsh(a))
            elif op == "svd":
                u, s, vh = sl.svd(a)
                arm["m"].update(u=m(u), vh=m(vh))
                arm["v"]["s"] = v(s)
                ut, st, vht = sl.svd(a, full_matrices=False)
                arm["m"].update(ut=m(ut), vht=m(vht))
                arm["v"]["svals"] = v(sl.svdvals(a))
            elif op == "lstsq":
                x, res, rank, s = sl.lstsq(a, b)
                arm["v"]["x"] = v(x)
                arm["v"]["sv"] = v(s)
                arm["s"]["rank"] = float(rank)
                arm["s"]["res"] = float(res) if np.size(res) == 1 else -1.0
            elif op == "pinv":
                arm["m"]["pinv"] = m(sl.pinv(a))
            elif op == "norm":
                for k, o in (("fro", "fro"), ("one", 1), ("inf", np.inf), ("two", 2)):
                    arm["s"][k] = float(sl.norm(a, o))
            elif op == "expm":
                arm["m"]["f"] = m(sl.expm(a))
            elif op == "sqrtm":
                arm["m"]["f"] = m(sl.sqrtm(a))
            elif op == "logm":
                arm["m"]["f"] = m(sl.logm(a))
            elif op in ("logm_real", "sqrtm_real"):
                ra = np.array(q["real"], dtype=float)
                f = sl.logm(ra) if op == "logm_real" else sl.sqrtm(ra)
                arm["cplx"] = bool(np.iscomplexobj(f))
                arm["m"]["f"] = m(f)
            arm["ok"] = True
    except Exception as e:
        arm["err"] = f"{type(e).__name__}: {e}"
    out.append(arm)
print(json.dumps(out))
"#;

fn run_oracle(query: &[Query]) -> Option<Vec<OracleArm>> {
    let payload = serde_json::to_string(query).expect("serialize complex-linalg query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(ORACLE)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn the complex-linalg oracle: {e}"
            );
            eprintln!("skipping complex-linalg oracle: {e}");
            return None;
        }
    };
    child
        .stdin
        .as_mut()
        .expect("oracle stdin")
        .write_all(payload.as_bytes())
        .expect("write oracle query");
    let output = child.wait_with_output().expect("wait for oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "complex-linalg oracle failed: {stderr}"
        );
        eprintln!("skipping complex-linalg oracle: scipy not available\n{stderr}");
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("parse oracle JSON"))
}

// ── comparison helpers ───────────────────────────────────────────────────

fn to_pairs(a: &[Vec<C64>]) -> Vec<Vec<Pair>> {
    a.iter()
        .map(|r| r.iter().map(|z| [z.re, z.im]).collect())
        .collect()
}

fn mat(p: &[Vec<Pair>]) -> ComplexMatrix {
    p.iter()
        .map(|r| r.iter().map(|q| C64::new(q[0], q[1])).collect())
        .collect()
}

fn vecc(p: &[Pair]) -> Vec<C64> {
    p.iter().map(|q| C64::new(q[0], q[1])).collect()
}

/// `max|x − y| / max(1, max|y|)`; a shape mismatch or a non-finite entry where SciPy's is finite
/// is +∞.
fn rel_err_mat(x: &[Vec<C64>], y: &[Vec<C64>]) -> f64 {
    if x.len() != y.len() || x.iter().zip(y).any(|(a, b)| a.len() != b.len()) {
        return f64::INFINITY;
    }
    rel_err_vec(
        &x.iter().flatten().copied().collect::<Vec<_>>(),
        &y.iter().flatten().copied().collect::<Vec<_>>(),
    )
}

fn rel_err_vec(x: &[C64], y: &[C64]) -> f64 {
    if x.len() != y.len() {
        return f64::INFINITY;
    }
    let scale = y.iter().map(|z| z.norm()).fold(1.0_f64, f64::max);
    let mut worst = 0.0_f64;
    for (a, b) in x.iter().zip(y) {
        let same_nonfinite =
            |p: f64, q: f64| (p.is_nan() && q.is_nan()) || (p.is_infinite() && p == q);
        let parts = [(a.re, b.re), (a.im, b.im)];
        for (p, q) in parts {
            if q.is_finite() {
                if !p.is_finite() {
                    return f64::INFINITY;
                }
            } else if !same_nonfinite(p, q) {
                return f64::INFINITY;
            }
        }
        if b.re.is_finite() && b.im.is_finite() {
            worst = worst.max((a - b).norm() / scale);
        }
    }
    worst
}

fn col(a: &[Vec<C64>], j: usize) -> Vec<C64> {
    a.iter().map(|r| r[j]).collect()
}

fn inner(x: &[C64], y: &[C64]) -> C64 {
    x.iter().zip(y).map(|(a, b)| a.conj() * b).sum()
}

/// `max_j | |x_jᴴ y_j| − 1 |` over the first `k` columns: unit-modulus phase freedom only.
fn phase_free_err(x: &[Vec<C64>], y: &[Vec<C64>], k: usize) -> f64 {
    (0..k)
        .map(|j| (inner(&col(x, j), &col(y, j)).norm() - 1.0).abs())
        .fold(0.0, f64::max)
}

fn projector(u: &[Vec<C64>], cols: std::ops::Range<usize>) -> ComplexMatrix {
    let m = u.len();
    (0..m)
        .map(|i| {
            (0..m)
                .map(|j| cols.clone().map(|c| u[i][c] * u[j][c].conj()).sum())
                .collect()
        })
        .collect()
}

fn opts() -> DecompOptions {
    DecompOptions::default()
}

/// Compare one case; `Ok(max_err)` when fsci produced a value (pass decided by the caller's
/// tolerance), `Err(reason)` when fsci failed.
fn compare(c: &Case, arm: &OracleArm) -> Result<(f64, f64), String> {
    let e = |err: fsci_linalg::LinalgError| format!("fsci error: {err}");
    let a = &c.a;
    let gm = |k: &str| mat(&arm.m[k]);
    let gv = |k: &str| vecc(&arm.v[k]);
    Ok(match c.op {
        "solve" => {
            let x = cx::solve(a, c.b.as_ref().unwrap(), SolveOptions::default()).map_err(e)?;
            (rel_err_vec(&x.x, &gv("x")), DIRECT_TOL)
        }
        "solve_pos" => {
            let o = SolveOptions {
                assume_a: Some(MatrixAssumption::PositiveDefinite),
                ..SolveOptions::default()
            };
            let x = cx::solve(a, c.b.as_ref().unwrap(), o).map_err(e)?;
            (rel_err_vec(&x.x, &gv("x")), DIRECT_TOL)
        }
        "lu" => {
            let r = cx::lu(a, opts()).map_err(e)?;
            let err = rel_err_mat(&cx::to_complex(&r.p), &gm("p"))
                .max(rel_err_mat(&r.l, &gm("l")))
                .max(rel_err_mat(&r.u, &gm("u")));
            (err, DIRECT_TOL)
        }
        "lu_solve" => {
            let f = cx::lu_factor(a, opts()).map_err(e)?;
            let piv: Vec<C64> = f.piv().iter().map(|&p| C64::new(p as f64, 0.0)).collect();
            let mut err = rel_err_mat(&f.lu(), &gm("lu")).max(rel_err_vec(&piv, &gv("piv")));
            for t in 0..=2u8 {
                let x = cx::lu_solve(&f, c.b.as_ref().unwrap(), t).map_err(e)?;
                err = err.max(rel_err_vec(&x, &gv(&format!("x{t}"))));
            }
            (err, DIRECT_TOL)
        }
        "det" => {
            let d = cx::det(a, opts()).map_err(e)?;
            (rel_err_vec(&[d], &gv("det")), DIRECT_TOL)
        }
        "inv" => (
            rel_err_mat(&cx::inv(a, opts()).map_err(e)?, &gm("inv")),
            DIRECT_TOL,
        ),
        "cholesky" => {
            let u = cx::cholesky(a, false, opts()).map_err(e)?;
            let l = cx::cholesky(a, true, opts()).map_err(e)?;
            (
                rel_err_mat(&u, &gm("u")).max(rel_err_mat(&l, &gm("l"))),
                DIRECT_TOL,
            )
        }
        "qr" | "qr_economic" => {
            let r = cx::qr(a, c.op == "qr_economic", opts()).map_err(e)?;
            (
                rel_err_mat(&r.q, &gm("q")).max(rel_err_mat(&r.r, &gm("r"))),
                DIRECT_TOL,
            )
        }
        "hessenberg" => {
            let r = cx::hessenberg(a, opts()).map_err(e)?;
            (
                rel_err_mat(&r.h, &gm("h")).max(rel_err_mat(&r.q, &gm("q"))),
                DIRECT_TOL,
            )
        }
        "schur" => {
            // A Schur pair is unique only up to a diagonal unitary D (Z' = Z·D, T' = Dᴴ·T·D)
            // once the eigenvalue order is fixed, and zlahqr's D is rounding noise: when a
            // subdiagonal converges to a few ulps, its phase rotates the column (measured:
            // reference LAPACK zlahqr called on fsci's own first iterate reproduces fsci's
            // phase, so the difference is the noise, not the algorithm). Compare the diagonal
            // positionally and the rest after estimating D from the columns.
            let r = cx::schur(a, opts()).map_err(e)?;
            let (st, sz) = (gm("t"), gm("z"));
            let n = a.len();
            let d: Vec<C64> = (0..n)
                .map(|j| {
                    let p = inner(&col(&r.z, j), &col(&sz, j));
                    p / p.norm()
                })
                .collect();
            let zd: ComplexMatrix =
                r.z.iter()
                    .map(|row| row.iter().zip(&d).map(|(z, dj)| z * dj).collect())
                    .collect();
            let tdd: ComplexMatrix = (0..n)
                .map(|i| (0..n).map(|j| d[i].conj() * r.t[i][j] * d[j]).collect())
                .collect();
            (
                rel_err_mat(&tdd, &st).max(rel_err_mat(&zd, &sz)),
                SPECTRAL_TOL,
            )
        }
        "eig" => {
            let r = cx::eig(a, opts()).map_err(e)?;
            let err =
                rel_err_vec(&r.eigenvalues, &gv("w")).max(rel_err_mat(&r.eigenvectors, &gm("v")));
            (err, SPECTRAL_TOL)
        }
        "eigvals" => (
            rel_err_vec(&cx::eigvals(a, opts()).map_err(e)?, &gv("w")),
            SPECTRAL_TOL,
        ),
        "eigh" => {
            let r = cx::eigh(a, opts()).map_err(e)?;
            let w: Vec<C64> = r.eigenvalues.iter().map(|&x| C64::new(x, 0.0)).collect();
            let wh: Vec<C64> = cx::eigvalsh(a, opts())
                .map_err(e)?
                .iter()
                .map(|&x| C64::new(x, 0.0))
                .collect();
            let n = a.len();
            let err = rel_err_vec(&w, &gv("w"))
                .max(rel_err_vec(&wh, &gv("wh")))
                .max(phase_free_err(&r.eigenvectors, &gm("v"), n));
            (err, SPECTRAL_TOL)
        }
        "svd" => {
            let (m, n) = (a.len(), a[0].len());
            let k = m.min(n);
            let full = cx::svd(a, true, opts()).map_err(e)?;
            let thin = cx::svd(a, false, opts()).map_err(e)?;
            let svals = cx::svdvals(a, opts()).map_err(e)?;
            let to_c = |s: &[f64]| s.iter().map(|&x| C64::new(x, 0.0)).collect::<Vec<_>>();
            let su = gm("u");
            let svh = gm("vh");
            let fv = cx::conj_transpose(&full.vh);
            let sv = cx::conj_transpose(&svh);
            let mut err = rel_err_vec(&to_c(&full.s), &gv("s"))
                .max(rel_err_vec(&to_c(&svals), &gv("svals")))
                .max(phase_free_err(&full.u, &su, k))
                .max(phase_free_err(&fv, &sv, k))
                .max(phase_free_err(&thin.u, &gm("ut"), k));
            // The left and right vectors of one triplet carry the same phase.
            for j in 0..k {
                let pu = inner(&col(&full.u, j), &col(&su, j));
                let pv = inner(&col(&fv, j), &col(&sv, j));
                err = err.max((pu - pv).norm());
            }
            // The completed columns span the same complement.
            if m > k {
                err = err.max(rel_err_mat(
                    &projector(&full.u, k..m),
                    &projector(&su, k..m),
                ));
            }
            if n > k {
                err = err.max(rel_err_mat(&projector(&fv, k..n), &projector(&sv, k..n)));
            }
            (err, SPECTRAL_TOL)
        }
        "lstsq" => {
            let r = cx::lstsq(a, c.b.as_ref().unwrap(), None, opts()).map_err(e)?;
            let s: Vec<C64> = r.s.iter().map(|&x| C64::new(x, 0.0)).collect();
            let mut err = rel_err_vec(&r.x, &gv("x")).max(rel_err_vec(&s, &gv("sv")));
            if (r.rank as f64 - arm.s["rank"]).abs() > 0.0 {
                err = f64::INFINITY;
            }
            let scipy_res = arm.s["res"];
            match r.residues {
                Some(res) => err = err.max((res - scipy_res).abs() / scipy_res.abs().max(1.0)),
                None if scipy_res >= 0.0 => err = f64::INFINITY,
                None => {}
            }
            (err, DIRECT_TOL)
        }
        "pinv" => (
            rel_err_mat(&cx::pinv(a, None, None, opts()).map_err(e)?, &gm("pinv")),
            SPECTRAL_TOL,
        ),
        "norm" => {
            let mut err = 0.0_f64;
            for (k, kind) in [
                ("fro", NormKind::Fro),
                ("one", NormKind::One),
                ("inf", NormKind::Inf),
                ("two", NormKind::Spectral),
            ] {
                let got = cx::norm(a, kind, opts()).map_err(e)?;
                let want = arm.s[k];
                err = err.max((got - want).abs() / want.abs().max(1.0));
            }
            (err, DIRECT_TOL)
        }
        "expm" => (
            rel_err_mat(&cx::expm(a, opts()).map_err(e)?, &gm("f")),
            SPECTRAL_TOL,
        ),
        "sqrtm" => (
            rel_err_mat(&cx::sqrtm(a, opts()).map_err(e)?, &gm("f")),
            SPECTRAL_TOL,
        ),
        "logm" => (
            rel_err_mat(&cx::logm(a, opts()).map_err(e)?, &gm("f")),
            SPECTRAL_TOL,
        ),
        "logm_real" | "sqrtm_real" => {
            let real = c.real.as_ref().unwrap();
            let r = if c.op == "logm_real" {
                cx::logm_real(real, opts())
            } else {
                cx::sqrtm_real(real, opts())
            }
            .map_err(e)?;
            if Some(r.is_complex()) != arm.cplx {
                return Err(format!(
                    "result dtype complex={} vs SciPy complex={:?}",
                    r.is_complex(),
                    arm.cplx
                ));
            }
            let got = match r {
                MaybeComplexMatrix::Real(m) => cx::to_complex(&m),
                MaybeComplexMatrix::Complex(m) => m,
            };
            (rel_err_mat(&got, &gm("f")), SPECTRAL_TOL)
        }
        other => return Err(format!("unknown op {other}")),
    })
}

#[test]
fn diff_linalg_complex_core() {
    let cases = cases();
    let query: Vec<Query> = cases
        .iter()
        .map(|c| Query {
            id: c.id.clone(),
            op: c.op.to_string(),
            a: to_pairs(&c.a),
            b: c.b
                .as_ref()
                .map(|b| b.iter().map(|z| [z.re, z.im]).collect()),
            real: c.real.clone(),
        })
        .collect();
    let Some(oracle) = run_oracle(&query) else {
        return;
    };
    let arms: HashMap<String, OracleArm> = oracle.into_iter().map(|a| (a.id.clone(), a)).collect();
    let mut ledger = CompareLedger::new("diff_linalg_complex_core", ARMS);
    let mut diffs = Vec::new();
    let mut max_err_by_op: BTreeMap<String, f64> = BTreeMap::new();
    for c in &cases {
        let arm = &arms[&c.id];
        let mut diff = CaseDiff {
            id: c.id.clone(),
            op: c.op.to_string(),
            max_err: f64::NAN,
            pass: false,
            reason: String::new(),
        };
        if !arm.ok {
            let reason = arm.err.clone().unwrap_or_default();
            ledger.oracle_missing(c.op, &c.id, &reason);
            diff.reason = format!("SciPy failed: {reason}");
            diffs.push(diff);
            continue;
        }
        match compare(c, arm) {
            Ok((err, tol)) => {
                diff.max_err = err;
                diff.pass = err <= tol;
                if !diff.pass {
                    diff.reason = format!("max rel err {err:e} > {tol:e}");
                }
                let slot = max_err_by_op.entry(c.op.to_string()).or_insert(0.0);
                *slot = slot.max(err);
                ledger.compared(c.op, &c.id, diff.pass);
            }
            Err(reason) => {
                ledger.rust_failed(c.op, &c.id, &reason);
                diff.reason = reason;
            }
        }
        diffs.push(diff);
    }
    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_linalg_complex_core".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_err_by_op: max_err_by_op.clone(),
        pass: all_pass,
        timestamp_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis()),
        cases: diffs.clone(),
    };
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("fixtures/artifacts/{PACKET_ID}/diff"));
    fs::create_dir_all(&dir).expect("create diff dir");
    fs::write(
        dir.join("diff_linalg_complex_core.json"),
        serde_json::to_string_pretty(&log).expect("serialize log"),
    )
    .expect("write diff log");
    for d in diffs.iter().filter(|d| !d.pass) {
        println!("FAIL {} ({}): {}", d.id, d.op, d.reason);
    }
    for (op, err) in &max_err_by_op {
        println!("{op:<12} max rel err {err:.3e}");
    }
    println!("{} cases compared", diffs.len());
    assert!(all_pass, "fsci_linalg::complex vs scipy.linalg failed");
    ledger.finish(1);
}
