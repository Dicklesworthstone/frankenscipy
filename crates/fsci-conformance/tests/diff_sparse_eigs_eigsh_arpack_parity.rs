#![forbid(unsafe_code)]
//! Live `scipy.sparse.linalg.eigsh` / `eigs` (ARPACK) parity for `fsci_sparse::eigsh` / `eigs`
//! (frankenscipy-1ksfv.10): restarted Krylov–Schur, every `which`, `sigma` shift-invert, the
//! generalized problem with a mass matrix, and honest non-convergence.
//!
//! Every case builds its matrix here, sends the triplets to SciPy 1.17.1, and compares:
//! - eigenvalues, sorted by (re, im): `|Δ| ≤ EIG_REL_TOL·max|λ|`;
//! - fsci's eigenpairs: `‖A x − λ M x‖ ≤ RESID_REL_TOL·‖A‖` (‖A‖ = max row/column abs sum), and
//!   for `eigsh` `max|XᵀMX − I| ≤ ORTHO_TOL`;
//! - `converged`: fsci must return `Ok` (whose `converged` is true) where SciPy converges, and
//!   refuse with the same class where SciPy raises: `EigsNoConvergence` for
//!   `ArpackNoConvergence`, `SingularMatrix` for "Factor is exactly singular".
//!
//! Matrices: 1-D Laplacians (n = 100, 2000, 20000) with closed-form spectra, a linear-FEM
//! stiffness/mass pair, normal nonsymmetric matrices with known complex spectra (a permuted
//! block diagonal of rotation-scaling blocks), 1-D and 2-D convection–diffusion operators, and a
//! clustered diagonal spectrum. k ∈ {1, 4, 6, 20}.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_sparse::{
    CooMatrix, CsrMatrix, EigsOptions, EigsResult, EigsWhich, FormatConvertible, Shape2D,
    SparseError, eigs, eigsh,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
/// Sorted eigenvalues against SciPy: `|Δ| ≤ EIG_REL_TOL·max|λ|` (the bead's acceptance bound).
const EIG_REL_TOL: f64 = 1.0e-9;
/// fsci's Ritz residuals: `‖A x − λ M x‖ ≤ RESID_REL_TOL·‖A‖`.
const RESID_REL_TOL: f64 = 1.0e-8;
/// `eigsh` eigenvectors: `max|XᵀMX − I| ≤ ORTHO_TOL`.
const ORTHO_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

type TestResult<T> = Result<T, String>;

#[derive(Debug, Clone, Default, Serialize)]
struct Triplets {
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<f64>,
}

impl Triplets {
    fn push(&mut self, row: usize, col: usize, val: f64) {
        self.rows.push(row);
        self.cols.push(col);
        self.vals.push(val);
    }

    fn csr(&self, n: usize) -> TestResult<CsrMatrix> {
        CooMatrix::from_triplets(
            Shape2D::new(n, n),
            self.vals.clone(),
            self.rows.clone(),
            self.cols.clone(),
            true,
        )
        .and_then(|coo| coo.to_csr())
        .map_err(|err| format!("build csr: {err}"))
    }
}

/// What SciPy is expected to do with a case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Expect {
    Values,
    NoConvergence,
    Singular,
}

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    func: &'static str,
    n: usize,
    a: Triplets,
    mass: Option<Triplets>,
    k: usize,
    which: &'static str,
    sigma: Option<f64>,
    maxiter: Option<usize>,
    expect: Expect,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleArm {
    case_id: String,
    status: String,
    re: Option<Vec<f64>>,
    im: Option<Vec<f64>>,
    message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
    n: usize,
    k: usize,
    which: String,
    sigma: Option<f64>,
    scipy_status: String,
    fsci_outcome: String,
    max_eig_diff: Option<f64>,
    max_rel_residual: Option<f64>,
    ortho_defect: Option<f64>,
    iterations: Option<usize>,
    nmatvec: Option<usize>,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    pass: bool,
    timestamp_ms: u128,
    elapsed_ms: u128,
    cases: Vec<CaseDiff>,
}

fn laplacian_1d(n: usize) -> Triplets {
    let mut t = Triplets::default();
    for i in 0..n {
        t.push(i, i, 2.0);
        if i + 1 < n {
            t.push(i, i + 1, -1.0);
            t.push(i + 1, i, -1.0);
        }
    }
    t
}

/// Linear FEM for −u'' = λu on (0, 1): stiffness tridiag(−1, 2, −1)/h, mass h·tridiag(1, 4, 1)/6.
fn fem_pair(n: usize) -> (Triplets, Triplets) {
    let h = 1.0 / (n + 1) as f64;
    let (mut k, mut m) = (Triplets::default(), Triplets::default());
    for i in 0..n {
        k.push(i, i, 2.0 / h);
        m.push(i, i, 4.0 * h / 6.0);
        if i + 1 < n {
            k.push(i, i + 1, -1.0 / h);
            k.push(i + 1, i, -1.0 / h);
            m.push(i, i + 1, h / 6.0);
            m.push(i + 1, i, h / 6.0);
        }
    }
    (k, m)
}

/// A fixed permutation of 0..n (an LCG Fisher–Yates; the seed is used by no fixture).
fn permutation(n: usize, seed: u64) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    let mut state = seed;
    for i in (1..n).rev() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        perm.swap(i, (state >> 33) as usize % (i + 1));
    }
    perm
}

/// A normal nonsymmetric matrix: 2×2 blocks [[a, b], [−b, a]] (eigenvalues a ± ib) for
/// `pair(j)`, plus real eigenvalues `reals`, scattered by a permutation similarity.
fn normal_blocks(
    pairs: usize,
    pair: impl Fn(usize) -> (f64, f64),
    reals: &[f64],
    seed: u64,
) -> Triplets {
    let n = 2 * pairs + reals.len();
    let perm = permutation(n, seed);
    let mut t = Triplets::default();
    for j in 0..pairs {
        let (a, b) = pair(j);
        let (p, q) = (perm[2 * j], perm[2 * j + 1]);
        t.push(p, p, a);
        t.push(p, q, b);
        t.push(q, p, -b);
        t.push(q, q, a);
    }
    for (i, &r) in reals.iter().enumerate() {
        let p = perm[2 * pairs + i];
        t.push(p, p, r);
    }
    t
}

/// 1-D convection–diffusion tridiag(−1 − c, 2, −1 + c).
fn convection_diffusion_1d(n: usize, c: f64) -> Triplets {
    let mut t = Triplets::default();
    for i in 0..n {
        t.push(i, i, 2.0);
        if i + 1 < n {
            t.push(i, i + 1, -1.0 + c);
            t.push(i + 1, i, -1.0 - c);
        }
    }
    t
}

/// 2-D convection–diffusion on a side×side grid (5-point diffusion, centred convection with
/// cell Péclet numbers cx, cy): mildly non-normal, real spectrum.
fn convection_diffusion_2d(side: usize, cx: f64, cy: f64) -> Triplets {
    let mut t = Triplets::default();
    let idx = |i: usize, j: usize| i + side * j;
    for j in 0..side {
        for i in 0..side {
            let p = idx(i, j);
            t.push(p, p, 4.0);
            if i > 0 {
                t.push(p, idx(i - 1, j), -1.0 - cx);
            }
            if i + 1 < side {
                t.push(p, idx(i + 1, j), -1.0 + cx);
            }
            if j > 0 {
                t.push(p, idx(i, j - 1), -1.0 - cy);
            }
            if j + 1 < side {
                t.push(p, idx(i, j + 1), -1.0 + cy);
            }
        }
    }
    t
}

fn diagonal(d: &[f64]) -> Triplets {
    let mut t = Triplets::default();
    for (i, &v) in d.iter().enumerate() {
        t.push(i, i, v);
    }
    t
}

fn case(
    case_id: &str,
    func: &'static str,
    n: usize,
    a: Triplets,
    mass: Option<Triplets>,
    k: usize,
    which: &'static str,
    sigma: Option<f64>,
    maxiter: Option<usize>,
    expect: Expect,
) -> Case {
    Case {
        case_id: case_id.to_string(),
        func,
        n,
        a,
        mass,
        k,
        which,
        sigma,
        maxiter,
        expect,
    }
}

fn generate_cases() -> Vec<Case> {
    use Expect::{NoConvergence, Singular, Values};
    let mut cases = Vec::new();
    let lap100 = laplacian_1d(100);
    // eigsh: every `which` on the 1-D Laplacian, k = 1, 6, 20.
    cases.push(case(
        "lap100_LM_k6",
        "eigsh",
        100,
        lap100.clone(),
        None,
        6,
        "LM",
        None,
        None,
        Values,
    ));
    cases.push(case(
        "lap100_SM_k6",
        "eigsh",
        100,
        lap100.clone(),
        None,
        6,
        "SM",
        None,
        None,
        Values,
    ));
    cases.push(case(
        "lap100_LA_k20",
        "eigsh",
        100,
        lap100.clone(),
        None,
        20,
        "LA",
        None,
        None,
        Values,
    ));
    cases.push(case(
        "lap100_SA_k1",
        "eigsh",
        100,
        lap100.clone(),
        None,
        1,
        "SA",
        None,
        None,
        Values,
    ));
    cases.push(case(
        "lap100_BE_k6",
        "eigsh",
        100,
        lap100,
        None,
        6,
        "BE",
        None,
        None,
        Values,
    ));
    // The bead's shift-invert and smallest-algebraic examples, n = 2000.
    let lap2000 = laplacian_1d(2000);
    cases.push(case(
        "lap2000_sigma1.0003_LM_k6",
        "eigsh",
        2000,
        lap2000.clone(),
        None,
        6,
        "LM",
        Some(1.0003),
        None,
        Values,
    ));
    cases.push(case(
        "lap2000_SA_k4",
        "eigsh",
        2000,
        lap2000.clone(),
        None,
        4,
        "SA",
        None,
        None,
        Values,
    ));
    cases.push(case(
        "lap20000_sigma0.5_LM_k6",
        "eigsh",
        20000,
        laplacian_1d(20000),
        None,
        6,
        "LM",
        Some(0.5),
        None,
        Values,
    ));
    // The generalized problem K x = λ M x (FEM stiffness/mass): shift-invert at 0 (mode 3 with
    // M) and M⁻¹K in the M inner product (mode 2).
    let (k2000, m2000) = fem_pair(2000);
    cases.push(case(
        "fem2000_M_sigma0_LM_k6",
        "eigsh",
        2000,
        k2000,
        Some(m2000),
        6,
        "LM",
        Some(0.0),
        None,
        Values,
    ));
    let (k100, m100) = fem_pair(100);
    cases.push(case(
        "fem100_M_LM_k6",
        "eigsh",
        100,
        k100,
        Some(m100),
        6,
        "LM",
        None,
        None,
        Values,
    ));

    // eigs: every `which` on normal matrices with known complex spectra.
    let reals: Vec<f64> = (0..10).map(|i| 0.2 + 0.13 * f64::from(i)).collect();
    let mixed = normal_blocks(
        55,
        |j| (1.0 + 0.9 * j as f64, 0.1 + 0.35 * (55 - j) as f64),
        &reals,
        0xA11C_E5EE_D001,
    );
    let pairs = normal_blocks(
        60,
        |j| (1.0 + 0.9 * j as f64, 0.1 + 0.35 * (60 - j) as f64),
        &[],
        0xA11C_E5EE_D002,
    );
    cases.push(case(
        "normal120_LM_k6",
        "eigs",
        120,
        mixed.clone(),
        None,
        6,
        "LM",
        None,
        None,
        Values,
    ));
    cases.push(case(
        "normal120_SM_k6",
        "eigs",
        120,
        mixed.clone(),
        None,
        6,
        "SM",
        None,
        None,
        Values,
    ));
    cases.push(case(
        "normal120_LR_k20",
        "eigs",
        120,
        mixed.clone(),
        None,
        20,
        "LR",
        None,
        None,
        Values,
    ));
    cases.push(case(
        "normal120_SR_k6",
        "eigs",
        120,
        mixed.clone(),
        None,
        6,
        "SR",
        None,
        None,
        Values,
    ));
    cases.push(case(
        "normal120_LI_k6",
        "eigs",
        120,
        mixed,
        None,
        6,
        "LI",
        None,
        None,
        Values,
    ));
    cases.push(case(
        "normal120_SI_k6",
        "eigs",
        120,
        pairs,
        None,
        6,
        "SI",
        None,
        None,
        Values,
    ));
    // Nonsymmetric convection–diffusion (real spectra).
    cases.push(case(
        "convdiff1d_100_LM_k1",
        "eigs",
        100,
        convection_diffusion_1d(100, 0.1),
        None,
        1,
        "LM",
        None,
        None,
        Values,
    ));
    cases.push(case(
        "convdiff1d_100_sigma1.5_k6",
        "eigs",
        100,
        convection_diffusion_1d(100, 0.1),
        None,
        6,
        "LM",
        Some(1.5),
        None,
        Values,
    ));
    let cd2 = convection_diffusion_2d(45, 0.05, 0.08);
    cases.push(case(
        "convdiff2d_2025_sigma0_k6",
        "eigs",
        2025,
        cd2.clone(),
        None,
        6,
        "LM",
        Some(0.0),
        None,
        Values,
    ));
    cases.push(case(
        "convdiff2d_2025_LR_k6",
        "eigs",
        2025,
        cd2,
        None,
        6,
        "LR",
        None,
        None,
        Values,
    ));
    // n = 20000 nonsymmetric: pairs with a small imaginary part near the shift.
    let big = normal_blocks(
        10000,
        |j| (j as f64, 0.01 + 1.0e-4 * j as f64),
        &[],
        0xA11C_E5EE_D003,
    );
    cases.push(case(
        "normal20000_sigma5000.3_k6",
        "eigs",
        20000,
        big,
        None,
        6,
        "LM",
        Some(5000.3),
        None,
        Values,
    ));

    // SciPy raises: no convergence within one or two restarts on a clustered spectrum, and a
    // shift exactly on an eigenvalue (2 − 2cos(667π/2001) = 1).
    let mut clustered: Vec<f64> = (0..10).map(|i| 1.0 + 1e-9 * f64::from(i)).collect();
    clustered.extend((0..190).map(|i| 0.01 + 0.89 * f64::from(i) / 189.0));
    cases.push(case(
        "clustered200_eigsh_maxiter1",
        "eigsh",
        200,
        diagonal(&clustered),
        None,
        6,
        "LM",
        None,
        Some(1),
        NoConvergence,
    ));
    cases.push(case(
        "clustered200_eigs_maxiter2",
        "eigs",
        200,
        diagonal(&clustered),
        None,
        6,
        "LM",
        None,
        Some(2),
        NoConvergence,
    ));
    cases.push(case(
        "lap2000_eigsh_sigma1.0_singular",
        "eigsh",
        2000,
        lap2000.clone(),
        None,
        6,
        "LM",
        Some(1.0),
        None,
        Singular,
    ));
    cases.push(case(
        "lap2000_eigs_sigma1.0_singular",
        "eigs",
        2000,
        lap2000,
        None,
        6,
        "LM",
        Some(1.0),
        None,
        Singular,
    ));
    cases
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn scipy_oracle_or_skip(cases: &[Case]) -> TestResult<Option<Vec<OracleArm>>> {
    let script = r#"
import json, sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, eigs, ArpackNoConvergence

out = []
for case in json.load(sys.stdin):
    arm = {"case_id": case["case_id"], "status": "error", "re": None, "im": None, "message": None}
    try:
        n = int(case["n"])
        t = case["a"]
        A = csr_matrix((t["vals"], (t["rows"], t["cols"])), shape=(n, n))
        M = None
        if case["mass"] is not None:
            m = case["mass"]
            M = csr_matrix((m["vals"], (m["rows"], m["cols"])), shape=(n, n))
        fn = eigsh if case["func"] == "eigsh" else eigs
        vals, _ = fn(A, k=int(case["k"]), M=M, sigma=case["sigma"], which=case["which"],
                     maxiter=case["maxiter"], return_eigenvectors=True)
        vals = np.asarray(vals)
        arm["re"] = [float(v) for v in np.real(vals)]
        arm["im"] = [float(v) for v in np.imag(vals)]
        arm["status"] = "ok"
    except ArpackNoConvergence as e:
        arm["status"] = "no_convergence"
        arm["message"] = str(e)
    except RuntimeError as e:
        arm["status"] = "singular" if "exactly singular" in str(e) else "error"
        arm["message"] = f"{type(e).__name__}: {e}"
    except Exception as e:
        arm["message"] = f"{type(e).__name__}: {e}"
    out.append(arm)
print(json.dumps(out))
"#;
    let query = serde_json::to_string(cases).map_err(|err| format!("serialize query: {err}"))?;
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
                return Err(format!(
                    "failed to spawn python3 for the ARPACK oracle: {err}"
                ));
            }
            eprintln!("skipping ARPACK oracle: python3 not available ({err})");
            return Ok(None);
        }
    };
    {
        let Some(stdin) = child.stdin.as_mut() else {
            return Err("open ARPACK oracle stdin".into());
        };
        if let Err(err) = stdin.write_all(query.as_bytes()) {
            let output = child
                .wait_with_output()
                .map_err(|wait_err| format!("wait for failed oracle: {wait_err}"))?;
            let stderr = String::from_utf8_lossy(&output.stderr);
            if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
                return Err(format!("ARPACK oracle stdin write failed: {err}; {stderr}"));
            }
            eprintln!("skipping ARPACK oracle: stdin write failed ({err})\n{stderr}");
            return Ok(None);
        }
    }
    let output = child
        .wait_with_output()
        .map_err(|err| format!("wait for ARPACK oracle: {err}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
            return Err(format!("ARPACK oracle failed: {stderr}"));
        }
        eprintln!("skipping ARPACK oracle: scipy not available\n{stderr}");
        return Ok(None);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout)
        .map(Some)
        .map_err(|err| format!("parse ARPACK oracle JSON: {err}"))
}

/// `max(‖A‖₁, ‖A‖∞)` from the triplets.
fn norm_bound(t: &Triplets, n: usize) -> f64 {
    let (mut rows, mut cols) = (vec![0.0_f64; n], vec![0.0_f64; n]);
    for ((&r, &c), &v) in t.rows.iter().zip(&t.cols).zip(&t.vals) {
        rows[r] += v.abs();
        cols[c] += v.abs();
    }
    rows.into_iter().chain(cols).fold(0.0_f64, f64::max)
}

fn matvec(a: &CsrMatrix, x: &[f64]) -> TestResult<Vec<f64>> {
    fsci_sparse::spmv_csr(a, x).map_err(|err| format!("spmv: {err}"))
}

/// `max_i ‖A x_i − λ_i M x_i‖` over the returned (complex) pairs.
fn max_residual(a: &CsrMatrix, mass: Option<&CsrMatrix>, r: &EigsResult) -> TestResult<f64> {
    let mut worst = 0.0_f64;
    for i in 0..r.eigenvalues.len() {
        let (lr, li) = (r.eigenvalues[i], r.eigenvalues_im[i]);
        let (xr, xi) = (&r.eigenvectors[i], &r.eigenvectors_im[i]);
        let (axr, axi) = (matvec(a, xr)?, matvec(a, xi)?);
        let (mxr, mxi) = match mass {
            Some(m) => (matvec(m, xr)?, matvec(m, xi)?),
            None => (xr.clone(), xi.clone()),
        };
        let mut sq = 0.0;
        for t in 0..xr.len() {
            let rr = axr[t] - (lr * mxr[t] - li * mxi[t]);
            let ri = axi[t] - (lr * mxi[t] + li * mxr[t]);
            sq += rr * rr + ri * ri;
        }
        // NaN propagates through max as NaN only via this explicit check.
        if sq.is_nan() {
            return Ok(f64::NAN);
        }
        worst = worst.max(sq.sqrt());
    }
    Ok(worst)
}

/// `max|XᵀMX − I|` over real eigenvectors.
fn ortho_defect(r: &EigsResult, mass: Option<&CsrMatrix>) -> TestResult<f64> {
    let mut worst = 0.0_f64;
    for (i, xi) in r.eigenvectors.iter().enumerate() {
        let bxi = match mass {
            Some(m) => matvec(m, xi)?,
            None => xi.clone(),
        };
        for (j, xj) in r.eigenvectors.iter().enumerate() {
            let dot: f64 = xj.iter().zip(&bxi).map(|(p, q)| p * q).sum();
            let target = if i == j { 1.0 } else { 0.0 };
            let d = (dot - target).abs();
            if d.is_nan() {
                return Ok(f64::NAN);
            }
            worst = worst.max(d);
        }
    }
    Ok(worst)
}

/// The maximum, NaN if any element is NaN (`f64::max` would drop it).
fn nan_max(values: impl Iterator<Item = f64>) -> f64 {
    let mut worst = 0.0_f64;
    for v in values {
        if v.is_nan() {
            return f64::NAN;
        }
        worst = worst.max(v);
    }
    worst
}

fn sorted_pairs(re: &[f64], im: &[f64]) -> Vec<(f64, f64)> {
    let mut v: Vec<(f64, f64)> = re.iter().copied().zip(im.iter().copied()).collect();
    v.sort_by(|x, y| x.0.total_cmp(&y.0).then_with(|| x.1.total_cmp(&y.1)));
    v
}

fn outcome_label(outcome: &Result<EigsResult, SparseError>) -> String {
    match outcome {
        Ok(r) => format!("ok(converged={})", r.converged),
        Err(SparseError::EigsNoConvergence { message, partial }) => {
            format!(
                "EigsNoConvergence({message}; {} partial)",
                partial.eigenvalues.len()
            )
        }
        Err(err) => format!("error({err})"),
    }
}

#[test]
fn diff_sparse_eigs_eigsh_arpack_parity() -> TestResult<()> {
    let start = Instant::now();
    let cases = generate_cases();
    let Some(oracle) = scipy_oracle_or_skip(&cases)? else {
        return Ok(());
    };
    let arms: HashMap<String, OracleArm> = oracle
        .into_iter()
        .map(|arm| (arm.case_id.clone(), arm))
        .collect();

    let mut ledger = CompareLedger::new(
        "diff_sparse_eigs_eigsh_arpack_parity",
        &["eigsh", "eigs", "raises"],
    );
    let mut diffs = Vec::with_capacity(cases.len());
    for case in &cases {
        let case_start = Instant::now();
        let a = case.a.csr(case.n)?;
        let mass = case.mass.as_ref().map(|m| m.csr(case.n)).transpose()?;
        let which: EigsWhich = case
            .which
            .parse()
            .map_err(|err: SparseError| format!("{}: {err}", case.case_id))?;
        let options = EigsOptions {
            which,
            sigma: case.sigma,
            mass: mass.as_ref(),
            max_iter: case.maxiter.unwrap_or(0),
            ..EigsOptions::default()
        };
        let outcome = if case.func == "eigsh" {
            eigsh(&a, case.k, options)
        } else {
            eigs(&a, case.k, options)
        };
        let scipy = arms.get(&case.case_id);
        let scipy_status = scipy.map_or_else(|| "missing".to_string(), |s| s.status.clone());
        let mut diff = CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.to_string(),
            n: case.n,
            k: case.k,
            which: case.which.to_string(),
            sigma: case.sigma,
            scipy_status: scipy_status.clone(),
            fsci_outcome: outcome_label(&outcome),
            max_eig_diff: None,
            max_rel_residual: None,
            ortho_defect: None,
            iterations: outcome.as_ref().ok().map(|r| r.iterations),
            nmatvec: outcome.as_ref().ok().map(|r| r.nmatvec),
            pass: false,
        };

        if case.expect != Expect::Values {
            let expected_status = if case.expect == Expect::NoConvergence {
                "no_convergence"
            } else {
                "singular"
            };
            if scipy_status != expected_status {
                // SciPy did not raise as this case says it does: the case is wrong, not fsci.
                ledger.oracle_missing(
                    "raises",
                    &case.case_id,
                    &format!(
                        "SciPy status {scipy_status} ({:?}), expected {expected_status}",
                        scipy.and_then(|s| s.message.clone())
                    ),
                );
            } else {
                let refused = match (&outcome, case.expect) {
                    (
                        Err(SparseError::EigsNoConvergence { partial, .. }),
                        Expect::NoConvergence,
                    ) => !partial.converged && partial.eigenvalues.len() < case.k,
                    (Err(SparseError::SingularMatrix { .. }), Expect::Singular) => true,
                    _ => false,
                };
                diff.pass = refused;
                ledger.expected_raise("raises", &case.case_id, refused);
            }
            println!(
                "{} scipy={} fsci={} pass={} ({:?})",
                case.case_id,
                scipy_status,
                diff.fsci_outcome,
                diff.pass,
                case_start.elapsed()
            );
            diffs.push(diff);
            continue;
        }

        let arm = case.func;
        let scipy_vals = scipy
            .filter(|s| s.status == "ok")
            .and_then(|s| Some(sorted_pairs(s.re.as_deref()?, s.im.as_deref()?)));
        let fsci_result = outcome.as_ref().ok().filter(|r| r.converged);
        let Some((want, result)) = ledger.both(arm, &case.case_id, scipy_vals, fsci_result) else {
            println!(
                "{} scipy={} ({:?}) fsci={} NOT COMPARED",
                case.case_id,
                scipy_status,
                scipy.and_then(|s| s.message.clone()),
                diff.fsci_outcome
            );
            diffs.push(diff);
            continue;
        };
        let got = sorted_pairs(&result.eigenvalues, &result.eigenvalues_im);
        let scale = want.iter().fold(0.0_f64, |acc, v| acc.max(v.0.hypot(v.1)));
        let eig_diff = if got.len() == want.len() {
            nan_max(
                got.iter()
                    .zip(&want)
                    .map(|(g, w)| (g.0 - w.0).hypot(g.1 - w.1)),
            )
        } else {
            f64::INFINITY
        };
        let a_norm = norm_bound(&case.a, case.n);
        let resid = max_residual(&a, mass.as_ref(), result)? / a_norm;
        let ortho = if case.func == "eigsh" {
            Some(ortho_defect(result, mass.as_ref())?)
        } else {
            None
        };
        let pass = eig_diff <= EIG_REL_TOL * scale
            && resid <= RESID_REL_TOL
            && ortho.is_none_or(|o| o <= ORTHO_TOL)
            && result.eigenvalues.len() == case.k;
        ledger.compared(arm, &case.case_id, pass);
        diff.max_eig_diff = Some(eig_diff);
        diff.max_rel_residual = Some(resid);
        diff.ortho_defect = ortho;
        diff.pass = pass;
        println!(
            "{} n={} k={} which={} sigma={:?}: iterations={} nmatvec={} max|dλ|={eig_diff:e} (bound {:e}) resid={resid:e} ortho={ortho:?} pass={pass} ({:?})",
            case.case_id,
            case.n,
            case.k,
            case.which,
            case.sigma,
            result.iterations,
            result.nmatvec,
            EIG_REL_TOL * scale,
            case_start.elapsed()
        );
        diffs.push(diff);
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_sparse_eigs_eigsh_arpack_parity".into(),
        category: "scipy.sparse.linalg.eigsh/eigs (ARPACK) vs fsci Krylov–Schur".into(),
        case_count: cases.len(),
        compared: ledger.counts().clone(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        elapsed_ms: start.elapsed().as_millis(),
        cases: diffs.clone(),
    };
    fs::create_dir_all(output_dir()).map_err(|err| format!("create diff dir: {err}"))?;
    fs::write(
        output_dir().join("diff_sparse_eigs_eigsh_arpack_parity.json"),
        serde_json::to_string_pretty(&log).map_err(|err| format!("serialize log: {err}"))?,
    )
    .map_err(|err| format!("write log: {err}"))?;

    let compared: usize = ledger.counts().values().map(|c| c.compared_cases).sum();
    println!(
        "compared {compared} of {} cases: {:?}",
        cases.len(),
        ledger.counts()
    );
    assert!(cases.len() >= 15, "the bead asks for at least 15 cases");
    assert_eq!(compared, cases.len(), "every case must be compared");
    assert!(all_pass, "an eigs/eigsh result disagrees with SciPy");
    ledger.finish(4);
    Ok(())
}
