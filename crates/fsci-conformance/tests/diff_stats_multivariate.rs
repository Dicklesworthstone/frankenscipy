#![forbid(unsafe_code)]
//! Live SciPy differential coverage for multivariate distributions:
//! - `MultivariateNormal`: pdf, logpdf, entropy, mahalanobis, cov, from_covariance
//! - `MultivariateT`: pdf, logpdf, cov, mahalanobis, from_covariance
//! - `Wishart`: pdf, logpdf, entropy, mean, from_covariance
//! - `InvWishart`: pdf, logpdf, mean, from_covariance
//!
//! Cross-checks FrankenSciPy against live SciPy reference oracle.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{
    Covariance, Dirichlet, DirichletMultinomial, InvWishart, MatrixNormal, MatrixT, Multinomial,
    MultivariateHypergeom, MultivariateNormal, MultivariateT, NormalInverseGamma, VonMisesFisher,
    Wishart, ortho_group, random_correlation, random_table, special_ortho_group, uniform_direction,
    unitary_group,
};
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

// Tolerance contracts. Every comparison below names one of these instead of
// passing a literal: G9's ratchet (`tolerance_lint`) reads `*_TOL` consts, and
// literal arguments to `check_pair` were invisible to it, so a 10x loosening at
// a call site passed the gate (zbtht.4). Tiers are `artifacts/TOLERANCE_POLICY.md` §1.

/// Closed-form moments, covariances, Mahalanobis distances, Cholesky entries,
/// and the Gaussian / Student-t densities: T3 rtol, atol one decade tighter.
const CLOSED_FORM_ABS_TOL: f64 = 1e-11;
const CLOSED_FORM_REL_TOL: f64 = 1e-10;
/// Densities and entropies built on multigammaln / digamma / Bessel series
/// (Wishart, InvWishart, MatrixNormal, MatrixT, VonMisesFisher, Dirichlet, and
/// the discrete pmfs): T4 ("special-function series") allows 1e-8.
const SPECIAL_FN_ABS_TOL: f64 = 1e-10;
const SPECIAL_FN_REL_TOL: f64 = 1e-9;
/// Entropies that sum a support or a Bessel-ratio series (VonMisesFisher,
/// Multinomial): T4.
const SERIES_ENTROPY_ABS_TOL: f64 = 1e-9;
const SERIES_ENTROPY_REL_TOL: f64 = 1e-8;
/// Worst deviation from unit norm of von Mises-Fisher samples, fsci vs SciPy: T4.
const UNIT_NORM_ERR_TOL: f64 = 1e-9;
/// SciPy's ortho_group / special_ortho_group determinant against 1: T4.
const ORTHO_DET_TOL: f64 = 1e-8;
/// Identities between two fsci constructions of one distribution
/// (`from_covariance` vs direct): a few roundings apart.
const FROM_COVARIANCE_IDENTITY_TOL: f64 = 1e-14;
/// Orthonormality / unitarity residual of a sampled matrix, fsci and SciPy.
const ORTHONORMALITY_TOL: f64 = 1e-10;
/// Unit diagonal of a random_correlation sample, fsci and SciPy: T5.
const CORRELATION_DIAG_TOL: f64 = 1e-7;
/// Sample means of fsci's seeded rvs (N = 1000-2000 draws) against the exact
/// mean. These check a sampler's consistency, not SciPy agreement, and they sit
/// ABOVE the policy's T6 ceiling (1e-1) without a §5 exception. They are named
/// here so they cannot loosen further; replacing them with a z-score bound is
/// frankenscipy-el4n5.
const MVHYPERGEOM_RVS_MEAN_TOL: f64 = 0.2;
const RVS_MEAN_TOL: f64 = 0.25;
const MATRIX_T_RVS_MEAN_TOL: f64 = 0.35;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiffRecord {
    case_id: String,
    family: String,
    rust_val: f64,
    scipy_val: f64,
    abs_diff: f64,
    rel_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    max_rel_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    records: Vec<DiffRecord>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn emit_log(log: &DiffLog) {
    let _ = fs::create_dir_all(output_dir());
    let path = output_dir().join(format!("{}.json", log.test_id));
    if let Ok(json) = serde_json::to_string_pretty(log) {
        let _ = fs::write(path, json);
    }
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

#[derive(Debug, Clone, Serialize)]
struct MvnCase {
    case_id: String,
    mean: Vec<f64>,
    cov: Vec<Vec<f64>>,
    x: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct MvtCase {
    case_id: String,
    loc: Vec<f64>,
    shape: Vec<Vec<f64>>,
    df: f64,
    x: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct WishartCase {
    case_id: String,
    df: f64,
    scale: Vec<Vec<f64>>,
    x: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct InvWishartCase {
    case_id: String,
    df: f64,
    scale: Vec<Vec<f64>>,
    x: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct MatrixNormalCase {
    case_id: String,
    mean: Vec<Vec<f64>>,
    rowcov: Vec<Vec<f64>>,
    colcov: Vec<Vec<f64>>,
    x: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct MatrixTCase {
    case_id: String,
    mean: Vec<Vec<f64>>,
    row_spread: Vec<Vec<f64>>,
    col_spread: Vec<Vec<f64>>,
    df: f64,
    x: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct VonMisesFisherCase {
    case_id: String,
    mu: Vec<f64>,
    kappa: f64,
    x: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct DirichletCase {
    case_id: String,
    alpha: Vec<f64>,
    x: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct MultivariateHypergeomCase {
    case_id: String,
    m: Vec<usize>,
    n: usize,
    x: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct NormalInverseGammaCase {
    case_id: String,
    mu: f64,
    lmbda: f64,
    a: f64,
    b: f64,
    x: f64,
    s2: f64,
}

#[derive(Debug, Clone, Serialize)]
struct MultinomialCase {
    case_id: String,
    n: usize,
    p: Vec<f64>,
    x: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct DirichletMultinomialCase {
    case_id: String,
    alpha: Vec<f64>,
    n: usize,
    x: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    mvn_cases: Vec<MvnCase>,
    mvt_cases: Vec<MvtCase>,
    wishart_cases: Vec<WishartCase>,
    invwishart_cases: Vec<InvWishartCase>,
    matrix_normal_cases: Vec<MatrixNormalCase>,
    matrix_t_cases: Vec<MatrixTCase>,
    vmf_cases: Vec<VonMisesFisherCase>,
    dirichlet_cases: Vec<DirichletCase>,
    mhypergeom_cases: Vec<MultivariateHypergeomCase>,
    nig_cases: Vec<NormalInverseGammaCase>,
    multinomial_cases: Vec<MultinomialCase>,
    dirichlet_multinomial_cases: Vec<DirichletMultinomialCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct MvnOracleResponse {
    case_id: String,
    pdf: f64,
    logpdf: f64,
    entropy: f64,
    mahalanobis_sq: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct MvtOracleResponse {
    case_id: String,
    pdf: f64,
    logpdf: f64,
    mahalanobis_sq: f64,
    cov_00: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct WishartOracleResponse {
    case_id: String,
    pdf: f64,
    logpdf: f64,
    entropy: f64,
    mean_00: f64,
    chol_00: f64,
    chol_10: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct InvWishartOracleResponse {
    case_id: String,
    pdf: f64,
    logpdf: f64,
    mean_00: Option<f64>,
    chol_00: f64,
    chol_10: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct MatrixNormalOracleResponse {
    case_id: String,
    pdf: f64,
    logpdf: f64,
    entropy: f64,
    rvs_shape: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct MatrixTOracleResponse {
    case_id: String,
    pdf: f64,
    logpdf: f64,
    rvs_shape: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct VonMisesFisherOracleResponse {
    case_id: String,
    pdf: f64,
    logpdf: f64,
    entropy: f64,
    rvs_shape: Vec<usize>,
    rvs_norm_err: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct DirichletOracleResponse {
    case_id: String,
    pdf: f64,
    logpdf: f64,
    mean: Vec<f64>,
    var: Vec<f64>,
    cov: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct MultivariateHypergeomOracleResponse {
    case_id: String,
    pmf: f64,
    logpmf: f64,
    mean: Vec<f64>,
    var: Vec<f64>,
    cov: Vec<Vec<f64>>,
    rvs_shape: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct NormalInverseGammaOracleResponse {
    case_id: String,
    pdf: f64,
    logpdf: f64,
    mean_x: f64,
    mean_s2: f64,
    var_x: f64,
    var_s2: f64,
    rvs_len: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct MultinomialOracleResponse {
    case_id: String,
    pmf: f64,
    logpmf: f64,
    mean: Vec<f64>,
    cov: Vec<Vec<f64>>,
    entropy: f64,
    rvs_shape: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct DirichletMultinomialOracleResponse {
    case_id: String,
    pmf: f64,
    logpmf: f64,
    mean: Vec<f64>,
    var: Vec<f64>,
    cov: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResponse {
    mvn: Vec<MvnOracleResponse>,
    mvt: Vec<MvtOracleResponse>,
    wishart: Vec<WishartOracleResponse>,
    invwishart: Vec<InvWishartOracleResponse>,
    matrix_normal: Vec<MatrixNormalOracleResponse>,
    matrix_t: Vec<MatrixTOracleResponse>,
    vmf: Vec<VonMisesFisherOracleResponse>,
    dirichlet: Vec<DirichletOracleResponse>,
    mhypergeom: Vec<MultivariateHypergeomOracleResponse>,
    nig: Vec<NormalInverseGammaOracleResponse>,
    multinomial: Vec<MultinomialOracleResponse>,
    dirichlet_multinomial: Vec<DirichletMultinomialOracleResponse>,
}

fn run_python_oracle(query: &OracleQuery) -> Option<OracleResponse> {
    let script = r#"
import json
import sys
import numpy as np
from scipy import stats

query = json.load(sys.stdin)
out = {
    "mvn": [],
    "mvt": [],
    "wishart": [],
    "invwishart": [],
    "matrix_normal": [],
    "matrix_t": [],
    "vmf": [],
    "dirichlet": [],
    "mhypergeom": [],
    "nig": [],
    "multinomial": [],
    "dirichlet_multinomial": [],
}

for c in query["mvn_cases"]:
    cid = c["case_id"]
    mean = np.array(c["mean"], dtype=np.float64)
    cov = np.array(c["cov"], dtype=np.float64)
    x = np.array(c["x"], dtype=np.float64)
    rv = stats.multivariate_normal(mean=mean, cov=cov)
    diff = x - mean
    inv_cov = np.linalg.inv(cov)
    maha_sq = float(diff.T @ inv_cov @ diff)
    out["mvn"].append({
        "case_id": cid,
        "pdf": float(rv.pdf(x)),
        "logpdf": float(rv.logpdf(x)),
        "entropy": float(rv.entropy()),
        "mahalanobis_sq": maha_sq,
    })

for c in query["mvt_cases"]:
    cid = c["case_id"]
    loc = np.array(c["loc"], dtype=np.float64)
    shape = np.array(c["shape"], dtype=np.float64)
    df = float(c["df"])
    x = np.array(c["x"], dtype=np.float64)
    rv = stats.multivariate_t(loc=loc, shape=shape, df=df)
    diff = x - loc
    inv_shape = np.linalg.inv(shape)
    maha_sq = float(diff.T @ inv_shape @ diff)
    cov_00 = float(shape[0, 0] * df / (df - 2.0)) if df > 2.0 else None
    out["mvt"].append({
        "case_id": cid,
        "pdf": float(rv.pdf(x)),
        "logpdf": float(rv.logpdf(x)),
        "mahalanobis_sq": maha_sq,
        "cov_00": cov_00,
    })

for c in query["wishart_cases"]:
    cid = c["case_id"]
    df = float(c["df"])
    scale = np.array(c["scale"], dtype=np.float64)
    x = np.array(c["x"], dtype=np.float64)
    rv = stats.wishart(df=df, scale=scale)
    m = rv.mean()
    c_mat = rv.C
    out["wishart"].append({
        "case_id": cid,
        "pdf": float(rv.pdf(x)),
        "logpdf": float(rv.logpdf(x)),
        "entropy": float(rv.entropy()),
        "mean_00": float(m[0, 0]),
        "chol_00": float(c_mat[0, 0]),
        "chol_10": float(c_mat[1, 0]),
    })

for c in query["invwishart_cases"]:
    cid = c["case_id"]
    df = float(c["df"])
    scale = np.array(c["scale"], dtype=np.float64)
    x = np.array(c["x"], dtype=np.float64)
    p = scale.shape[0]
    rv = stats.invwishart(df=df, scale=scale)
    c_mat = rv.C
    mean_00 = float(scale[0, 0] / (df - p - 1)) if df > p + 1 else None
    out["invwishart"].append({
        "case_id": cid,
        "pdf": float(rv.pdf(x)),
        "logpdf": float(rv.logpdf(x)),
        "mean_00": mean_00,
        "chol_00": float(c_mat[0, 0]),
        "chol_10": float(c_mat[1, 0]),
    })

for c in query["matrix_normal_cases"]:
    cid = c["case_id"]
    mean = np.array(c["mean"], dtype=np.float64)
    rowcov = np.array(c["rowcov"], dtype=np.float64)
    colcov = np.array(c["colcov"], dtype=np.float64)
    x = np.array(c["x"], dtype=np.float64)
    rv = stats.matrix_normal(mean=mean, rowcov=rowcov, colcov=colcov)
    rvs_samp = rv.rvs(size=50, random_state=42)
    out["matrix_normal"].append({
        "case_id": cid,
        "pdf": float(rv.pdf(x)),
        "logpdf": float(rv.logpdf(x)),
        "entropy": float(rv.entropy()),
        "rvs_shape": list(rvs_samp.shape),
    })

for c in query["matrix_t_cases"]:
    cid = c["case_id"]
    mean = np.array(c["mean"], dtype=np.float64)
    row_spread = np.array(c["row_spread"], dtype=np.float64)
    col_spread = np.array(c["col_spread"], dtype=np.float64)
    df = float(c["df"])
    x = np.array(c["x"], dtype=np.float64)
    rv = stats.matrix_t(mean=mean, row_spread=row_spread, col_spread=col_spread, df=df)
    rvs_samp = rv.rvs(size=50, random_state=42)
    out["matrix_t"].append({
        "case_id": cid,
        "pdf": float(rv.pdf(x)),
        "logpdf": float(rv.logpdf(x)),
        "rvs_shape": list(rvs_samp.shape),
    })

for c in query["vmf_cases"]:
    cid = c["case_id"]
    mu = np.array(c["mu"], dtype=np.float64)
    kappa = float(c["kappa"])
    x = np.array(c["x"], dtype=np.float64)
    rv = stats.vonmises_fisher(mu, kappa)
    rvs_samp = rv.rvs(size=50, random_state=42)
    vmf_norm_err = float(np.max(np.abs(np.linalg.norm(rvs_samp, axis=-1) - 1.0)))
    out["vmf"].append({
        "case_id": cid,
        "pdf": float(rv.pdf(x)),
        "logpdf": float(rv.logpdf(x)),
        "entropy": float(rv.entropy()),
        "rvs_shape": list(rvs_samp.shape),
        "rvs_norm_err": vmf_norm_err,
    })

for c in query["dirichlet_cases"]:
    cid = c["case_id"]
    alpha = np.array(c["alpha"], dtype=np.float64)
    x = np.array(c["x"], dtype=np.float64)
    rv = stats.dirichlet(alpha)
    cov_mat = [[float(v) for v in row] for row in rv.cov()]
    out["dirichlet"].append({
        "case_id": cid,
        "pdf": float(rv.pdf(x)),
        "logpdf": float(rv.logpdf(x)),
        "mean": [float(m) for m in rv.mean()],
        "var": [float(v) for v in rv.var()],
        "cov": cov_mat,
    })

for c in query["mhypergeom_cases"]:
    cid = c["case_id"]
    m = [int(v) for v in c["m"]]
    n = int(c["n"])
    x = [int(v) for v in c["x"]]
    rv = stats.multivariate_hypergeom(m=m, n=n)
    cov_mat = [[float(v) for v in row] for row in rv.cov()]
    rvs_samp = rv.rvs(size=50, random_state=42)
    out["mhypergeom"].append({
        "case_id": cid,
        "pmf": float(rv.pmf(x)),
        "logpmf": float(rv.logpmf(x)),
        "mean": [float(m) for m in rv.mean()],
        "var": [float(v) for v in rv.var()],
        "cov": cov_mat,
        "rvs_shape": list(rvs_samp.shape),
    })

for c in query["nig_cases"]:
    cid = c["case_id"]
    mu = float(c["mu"])
    lmbda = float(c["lmbda"])
    a = float(c["a"])
    b = float(c["b"])
    x = float(c["x"])
    s2 = float(c["s2"])
    rv = stats.normal_inverse_gamma(mu=mu, lmbda=lmbda, a=a, b=b)
    m = rv.mean()
    v = rv.var()
    rvs_samp = rv.rvs(size=50, random_state=42)
    rvs_len = int(len(rvs_samp[0]))
    out["nig"].append({
        "case_id": cid,
        "pdf": float(rv.pdf(x, s2)),
        "logpdf": float(rv.logpdf(x, s2)),
        "mean_x": float(m[0]),
        "mean_s2": float(m[1]),
        "var_x": float(v[0]),
        "var_s2": float(v[1]),
        "rvs_len": rvs_len,
    })

for c in query["multinomial_cases"]:
    cid = c["case_id"]
    n = int(c["n"])
    p = np.array(c["p"], dtype=np.float64)
    x = np.array(c["x"], dtype=np.float64)
    rv = stats.multinomial(n=n, p=p)
    cov_mat = [[float(val) for val in row] for row in rv.cov()]
    rvs_samp = rv.rvs(size=50, random_state=42)
    out["multinomial"].append({
        "case_id": cid,
        "pmf": float(rv.pmf(x)),
        "logpmf": float(rv.logpmf(x)),
        "mean": [float(m) for m in rv.mean()],
        "cov": cov_mat,
        "entropy": float(rv.entropy()),
        "rvs_shape": list(rvs_samp.shape),
    })

for c in query["dirichlet_multinomial_cases"]:
    cid = c["case_id"]
    alpha = np.array(c["alpha"], dtype=np.float64)
    n = int(c["n"])
    x = np.array(c["x"], dtype=np.float64)
    rv = stats.dirichlet_multinomial(alpha=alpha, n=n)
    cov_mat = [[float(val) for val in row] for row in rv.cov()]
    out["dirichlet_multinomial"].append({
        "case_id": cid,
        "pmf": float(rv.pmf(x)),
        "logpmf": float(rv.logpmf(x)),
        "mean": [float(m) for m in rv.mean()],
        "var": [float(v) for v in rv.var()],
        "cov": cov_mat,
    })

json.dump(out, sys.stdout)
"#;

    let mut child = fsci_conformance::scipy_oracle_command()
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    {
        let stdin = child.stdin.as_mut()?;
        let json_bytes = serde_json::to_vec(query).ok()?;
        stdin.write_all(&json_bytes).ok()?;
    }

    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }

    serde_json::from_slice(&output.stdout).ok()
}

fn check_pair(
    case_id: &str,
    family: &str,
    rust_val: f64,
    scipy_val: f64,
    atol: f64,
    rtol: f64,
    records: &mut Vec<DiffRecord>,
) {
    let abs_diff = (rust_val - scipy_val).abs();
    let rel_diff = if scipy_val.abs() > 1e-15 {
        abs_diff / scipy_val.abs()
    } else {
        abs_diff
    };
    let pass = abs_diff <= atol || rel_diff <= rtol;
    records.push(DiffRecord {
        case_id: case_id.to_string(),
        family: family.to_string(),
        rust_val,
        scipy_val,
        abs_diff,
        rel_diff,
        pass,
    });
    assert!(
        pass,
        "[{family}] {case_id}: rust={rust_val}, scipy={scipy_val}, abs_diff={abs_diff}, rel_diff={rel_diff} (atol={atol}, rtol={rtol})"
    );
}

#[test]
fn diff_multivariate_stats_scipy_oracle() {
    let t0 = Instant::now();

    // 1. Build test cases across dimensions
    let mvn_cases = vec![
        MvnCase {
            case_id: "mvn_2d_diag".into(),
            mean: vec![0.5, -1.0],
            cov: vec![vec![2.0, 0.0], vec![0.0, 1.5]],
            x: vec![1.0, -0.5],
        },
        MvnCase {
            case_id: "mvn_2d_rotated".into(),
            mean: vec![1.0, 2.0],
            cov: vec![vec![2.0, 0.6], vec![0.6, 1.0]],
            x: vec![1.5, 2.5],
        },
        MvnCase {
            case_id: "mvn_3d_corr".into(),
            mean: vec![0.0, 1.0, 2.0],
            cov: vec![
                vec![3.0, 0.5, 0.2],
                vec![0.5, 2.0, 0.4],
                vec![0.2, 0.4, 1.5],
            ],
            x: vec![0.2, 1.2, 1.8],
        },
    ];

    let mvt_cases = vec![
        MvtCase {
            case_id: "mvt_2d_df3".into(),
            loc: vec![0.0, 0.0],
            shape: vec![vec![1.5, 0.3], vec![0.3, 2.0]],
            df: 3.0,
            x: vec![0.5, -0.5],
        },
        MvtCase {
            case_id: "mvt_2d_df10".into(),
            loc: vec![1.0, -1.0],
            shape: vec![vec![2.0, 0.5], vec![0.5, 1.2]],
            df: 10.0,
            x: vec![1.2, -0.8],
        },
        MvtCase {
            case_id: "mvt_3d_df5".into(),
            loc: vec![0.5, 0.5, 0.5],
            shape: vec![
                vec![2.0, 0.2, 0.1],
                vec![0.2, 1.5, 0.3],
                vec![0.1, 0.3, 1.8],
            ],
            df: 5.0,
            x: vec![1.0, 0.0, 0.8],
        },
        MvtCase {
            case_id: "mvt_2d_df1_5_no_cov".into(),
            loc: vec![0.0, 0.0],
            shape: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            df: 1.5,
            x: vec![0.1, 0.2],
        },
    ];

    let wishart_cases = vec![
        WishartCase {
            case_id: "wishart_2d_df5".into(),
            df: 5.0,
            scale: vec![vec![1.5, 0.4], vec![0.4, 2.0]],
            x: vec![vec![3.0, 0.8], vec![0.8, 4.0]],
        },
        WishartCase {
            case_id: "wishart_3d_df8".into(),
            df: 8.0,
            scale: vec![
                vec![2.0, 0.3, 0.1],
                vec![0.3, 1.5, 0.2],
                vec![0.1, 0.2, 1.8],
            ],
            x: vec![
                vec![5.0, 0.7, 0.3],
                vec![0.7, 4.0, 0.5],
                vec![0.3, 0.5, 4.5],
            ],
        },
    ];

    let invwishart_cases = vec![
        InvWishartCase {
            case_id: "invwishart_2d_df6".into(),
            df: 6.0,
            scale: vec![vec![1.5, 0.4], vec![0.4, 2.0]],
            x: vec![vec![0.8, 0.1], vec![0.1, 1.0]],
        },
        InvWishartCase {
            case_id: "invwishart_3d_df9".into(),
            df: 9.0,
            scale: vec![
                vec![2.0, 0.3, 0.1],
                vec![0.3, 1.5, 0.2],
                vec![0.1, 0.2, 1.8],
            ],
            x: vec![
                vec![0.5, 0.05, 0.02],
                vec![0.05, 0.4, 0.03],
                vec![0.02, 0.03, 0.6],
            ],
        },
    ];

    let matrix_normal_cases = vec![
        MatrixNormalCase {
            case_id: "matrix_normal_2x2".into(),
            mean: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            rowcov: vec![vec![2.0, 0.3], vec![0.3, 1.0]],
            colcov: vec![vec![1.0, 0.2], vec![0.2, 1.5]],
            x: vec![vec![1.5, 2.5], vec![2.5, 3.5]],
        },
        MatrixNormalCase {
            case_id: "matrix_normal_3x2".into(),
            mean: vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
            rowcov: vec![
                vec![2.0, 0.1, 0.0],
                vec![0.1, 1.0, 0.2],
                vec![0.0, 0.2, 1.5],
            ],
            colcov: vec![vec![1.0, 0.3], vec![0.3, 2.0]],
            x: vec![vec![1.5, 2.5], vec![2.5, 3.5], vec![4.5, 7.0]],
        },
    ];

    let matrix_t_cases = vec![
        MatrixTCase {
            case_id: "matrix_t_2x2_df5".into(),
            mean: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            row_spread: vec![vec![2.0, 0.3], vec![0.3, 1.0]],
            col_spread: vec![vec![1.0, 0.2], vec![0.2, 1.5]],
            df: 5.0,
            x: vec![vec![1.5, 2.5], vec![2.5, 3.5]],
        },
        MatrixTCase {
            case_id: "matrix_t_2x3_df8".into(),
            mean: vec![vec![0.5, 1.0, -0.5], vec![1.2, 0.0, 2.1]],
            row_spread: vec![vec![1.5, 0.2], vec![0.2, 1.2]],
            col_spread: vec![
                vec![2.0, 0.1, 0.0],
                vec![0.1, 1.0, 0.3],
                vec![0.0, 0.3, 1.5],
            ],
            df: 8.0,
            x: vec![vec![0.6, 0.9, -0.4], vec![1.1, 0.1, 2.0]],
        },
    ];

    let vmf_cases = vec![
        VonMisesFisherCase {
            case_id: "vmf_3d_kappa5".into(),
            mu: vec![0.0, 0.0, 1.0],
            kappa: 5.0,
            x: vec![0.0, 0.6, 0.8],
        },
        VonMisesFisherCase {
            case_id: "vmf_4d_kappa2".into(),
            mu: vec![0.5, 0.5, 0.5, 0.5],
            kappa: 2.0,
            x: vec![0.0, 0.0, 0.0, 1.0],
        },
    ];

    let dirichlet_cases = vec![
        DirichletCase {
            case_id: "dirichlet_3d".into(),
            alpha: vec![1.0, 2.0, 3.0],
            x: vec![0.2, 0.3, 0.5],
        },
        DirichletCase {
            case_id: "dirichlet_4d".into(),
            alpha: vec![2.0, 1.5, 0.5, 3.0],
            x: vec![0.25, 0.2, 0.15, 0.4],
        },
    ];

    let mhypergeom_cases = vec![
        MultivariateHypergeomCase {
            case_id: "mhypergeom_3d".into(),
            m: vec![10, 8, 6],
            n: 5,
            x: vec![2, 2, 1],
        },
        MultivariateHypergeomCase {
            case_id: "mhypergeom_4d".into(),
            m: vec![5, 4, 3, 2],
            n: 4,
            x: vec![2, 1, 1, 0],
        },
    ];

    let nig_cases = vec![
        NormalInverseGammaCase {
            case_id: "nig_case1".into(),
            mu: 1.0,
            lmbda: 2.0,
            a: 3.0,
            b: 4.0,
            x: 1.5,
            s2: 2.0,
        },
        NormalInverseGammaCase {
            case_id: "nig_case2".into(),
            mu: -0.5,
            lmbda: 1.5,
            a: 4.0,
            b: 2.5,
            x: -0.2,
            s2: 1.0,
        },
        NormalInverseGammaCase {
            case_id: "nig_case3".into(),
            mu: 0.0,
            lmbda: 0.5,
            a: 5.0,
            b: 1.0,
            x: 0.8,
            s2: 0.5,
        },
    ];

    let multinomial_cases = vec![
        MultinomialCase {
            case_id: "multinomial_3d".into(),
            n: 10,
            p: vec![0.2, 0.3, 0.5],
            x: vec![2.0, 3.0, 5.0],
        },
        MultinomialCase {
            case_id: "multinomial_4d".into(),
            n: 20,
            p: vec![0.1, 0.2, 0.3, 0.4],
            x: vec![3.0, 4.0, 6.0, 7.0],
        },
        MultinomialCase {
            case_id: "multinomial_equal_p".into(),
            n: 12,
            p: vec![0.25, 0.25, 0.25, 0.25],
            x: vec![3.0, 3.0, 3.0, 3.0],
        },
    ];

    let dirichlet_multinomial_cases = vec![
        DirichletMultinomialCase {
            case_id: "dir_multinomial_3d".into(),
            alpha: vec![1.0, 2.0, 3.0],
            n: 6,
            x: vec![1.0, 2.0, 3.0],
        },
        DirichletMultinomialCase {
            case_id: "dir_multinomial_4d".into(),
            alpha: vec![2.0, 1.5, 3.5, 1.0],
            n: 10,
            x: vec![2.0, 1.0, 5.0, 2.0],
        },
        DirichletMultinomialCase {
            case_id: "dir_multinomial_fractional_alpha".into(),
            alpha: vec![0.5, 0.5, 1.0],
            n: 8,
            x: vec![1.0, 2.0, 5.0],
        },
    ];

    let query = OracleQuery {
        mvn_cases: mvn_cases.clone(),
        mvt_cases: mvt_cases.clone(),
        wishart_cases: wishart_cases.clone(),
        invwishart_cases: invwishart_cases.clone(),
        matrix_normal_cases: matrix_normal_cases.clone(),
        matrix_t_cases: matrix_t_cases.clone(),
        vmf_cases: vmf_cases.clone(),
        dirichlet_cases: dirichlet_cases.clone(),
        mhypergeom_cases: mhypergeom_cases.clone(),
        nig_cases: nig_cases.clone(),
        multinomial_cases: multinomial_cases.clone(),
        dirichlet_multinomial_cases: dirichlet_multinomial_cases.clone(),
    };

    let oracle_opt = run_python_oracle(&query);
    if oracle_opt.is_none() {
        if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
            panic!("SciPy oracle required via {REQUIRE_SCIPY_ENV} but unavailable or failed");
        }
        eprintln!("Skipping live differential tests: SciPy oracle not available");
        return;
    }
    let oracle = oracle_opt.unwrap();

    let mut records = Vec::new();

    // Test MultivariateNormal
    for (case, resp) in mvn_cases.iter().zip(oracle.mvn.iter()) {
        assert_eq!(case.case_id, resp.case_id);
        let dist = MultivariateNormal::new(&case.mean, &case.cov).expect("mvn new");

        // pdf & logpdf
        let rust_pdf = dist.pdf(&case.x).expect("mvn pdf");
        let rust_logpdf = dist.logpdf(&case.x).expect("mvn logpdf");
        check_pair(
            &format!("{}_pdf", case.case_id),
            "MultivariateNormal",
            rust_pdf,
            resp.pdf,
            CLOSED_FORM_ABS_TOL,
            CLOSED_FORM_REL_TOL,
            &mut records,
        );
        check_pair(
            &format!("{}_logpdf", case.case_id),
            "MultivariateNormal",
            rust_logpdf,
            resp.logpdf,
            CLOSED_FORM_ABS_TOL,
            CLOSED_FORM_REL_TOL,
            &mut records,
        );

        // entropy
        let rust_entropy = dist.entropy();
        check_pair(
            &format!("{}_entropy", case.case_id),
            "MultivariateNormal",
            rust_entropy,
            resp.entropy,
            CLOSED_FORM_ABS_TOL,
            CLOSED_FORM_REL_TOL,
            &mut records,
        );

        // mahalanobis squared
        let rust_maha_sq = dist.mahalanobis_squared(&case.x).expect("mvn maha_sq");
        check_pair(
            &format!("{}_maha_sq", case.case_id),
            "MultivariateNormal",
            rust_maha_sq,
            resp.mahalanobis_sq,
            CLOSED_FORM_ABS_TOL,
            CLOSED_FORM_REL_TOL,
            &mut records,
        );
        let rust_maha = dist.mahalanobis(&case.x).expect("mvn maha");
        check_pair(
            &format!("{}_maha", case.case_id),
            "MultivariateNormal",
            rust_maha,
            resp.mahalanobis_sq.sqrt(),
            CLOSED_FORM_ABS_TOL,
            CLOSED_FORM_REL_TOL,
            &mut records,
        );

        // Test from_covariance parity
        let cov_rep = Covariance::from_psd(&case.cov).expect("cov psd");
        let dist_from_cov =
            MultivariateNormal::from_covariance(&case.mean, &cov_rep).expect("mvn from_cov");
        assert_eq!(dist.dim(), dist_from_cov.dim());
        assert_eq!(dist.mean(), dist_from_cov.mean());
        assert_eq!(dist.cov(), dist_from_cov.cov());
        assert!((dist.entropy() - dist_from_cov.entropy()).abs() < FROM_COVARIANCE_IDENTITY_TOL);
        assert!(
            (dist.logpdf(&case.x).unwrap() - dist_from_cov.logpdf(&case.x).unwrap()).abs()
                < FROM_COVARIANCE_IDENTITY_TOL
        );
    }

    // Test MultivariateT
    for (case, resp) in mvt_cases.iter().zip(oracle.mvt.iter()) {
        assert_eq!(case.case_id, resp.case_id);
        let dist = MultivariateT::new(&case.loc, &case.shape, case.df).expect("mvt new");

        // pdf & logpdf
        let rust_pdf = dist.pdf(&case.x).expect("mvt pdf");
        let rust_logpdf = dist.logpdf(&case.x).expect("mvt logpdf");
        check_pair(
            &format!("{}_pdf", case.case_id),
            "MultivariateT",
            rust_pdf,
            resp.pdf,
            CLOSED_FORM_ABS_TOL,
            CLOSED_FORM_REL_TOL,
            &mut records,
        );
        check_pair(
            &format!("{}_logpdf", case.case_id),
            "MultivariateT",
            rust_logpdf,
            resp.logpdf,
            CLOSED_FORM_ABS_TOL,
            CLOSED_FORM_REL_TOL,
            &mut records,
        );

        // mahalanobis squared
        let rust_maha_sq = dist.mahalanobis_squared(&case.x).expect("mvt maha_sq");
        check_pair(
            &format!("{}_maha_sq", case.case_id),
            "MultivariateT",
            rust_maha_sq,
            resp.mahalanobis_sq,
            CLOSED_FORM_ABS_TOL,
            CLOSED_FORM_REL_TOL,
            &mut records,
        );

        // cov check
        if let Some(scipy_c00) = resp.cov_00 {
            let rust_cov = dist.cov().expect("mvt cov should exist for df > 2");
            check_pair(
                &format!("{}_cov_00", case.case_id),
                "MultivariateT",
                rust_cov[0][0],
                scipy_c00,
                CLOSED_FORM_ABS_TOL,
                CLOSED_FORM_REL_TOL,
                &mut records,
            );
        } else {
            assert!(dist.cov().is_none(), "mvt cov must be None for df <= 2");
        }

        // Test from_covariance parity
        let cov_rep = Covariance::from_psd(&case.shape).expect("cov psd");
        let dist_from_cov =
            MultivariateT::from_covariance(&case.loc, &cov_rep, case.df).expect("mvt from_cov");
        assert_eq!(dist.dim(), dist_from_cov.dim());
        assert_eq!(dist.loc(), dist_from_cov.loc());
        assert_eq!(dist.shape(), dist_from_cov.shape());
        assert_eq!(dist.df(), dist_from_cov.df());
        assert!(
            (dist.logpdf(&case.x).unwrap() - dist_from_cov.logpdf(&case.x).unwrap()).abs()
                < FROM_COVARIANCE_IDENTITY_TOL
        );
    }

    // Test Wishart
    for (case, resp) in wishart_cases.iter().zip(oracle.wishart.iter()) {
        assert_eq!(case.case_id, resp.case_id);
        let dist = Wishart::new(case.df, &case.scale).expect("wishart new");

        // pdf & logpdf
        let rust_pdf = dist.pdf(&case.x).expect("wishart pdf");
        let rust_logpdf = dist.logpdf(&case.x).expect("wishart logpdf");
        check_pair(
            &format!("{}_pdf", case.case_id),
            "Wishart",
            rust_pdf,
            resp.pdf,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );
        check_pair(
            &format!("{}_logpdf", case.case_id),
            "Wishart",
            rust_logpdf,
            resp.logpdf,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );

        // entropy
        let rust_entropy = dist.entropy();
        check_pair(
            &format!("{}_entropy", case.case_id),
            "Wishart",
            rust_entropy,
            resp.entropy,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );

        // mean
        let rust_mean = dist.mean();
        check_pair(
            &format!("{}_mean_00", case.case_id),
            "Wishart",
            rust_mean[0][0],
            resp.mean_00,
            CLOSED_FORM_ABS_TOL,
            CLOSED_FORM_REL_TOL,
            &mut records,
        );

        // c and C accessors
        let rust_chol = dist.c();
        assert_eq!(rust_chol, dist.C());
        check_pair(
            &format!("{}_chol_00", case.case_id),
            "Wishart",
            rust_chol[0][0],
            resp.chol_00,
            CLOSED_FORM_ABS_TOL,
            CLOSED_FORM_REL_TOL,
            &mut records,
        );
        check_pair(
            &format!("{}_chol_10", case.case_id),
            "Wishart",
            rust_chol[1][0],
            resp.chol_10,
            CLOSED_FORM_ABS_TOL,
            CLOSED_FORM_REL_TOL,
            &mut records,
        );

        // Test from_covariance parity
        let cov_rep = Covariance::from_psd(&case.scale).expect("cov psd");
        let dist_from_cov = Wishart::from_covariance(case.df, &cov_rep).expect("wishart from_cov");
        assert_eq!(dist.dim(), dist_from_cov.dim());
        assert_eq!(dist.df(), dist_from_cov.df());
        assert_eq!(dist.scale(), dist_from_cov.scale());
        assert!((dist.entropy() - dist_from_cov.entropy()).abs() < FROM_COVARIANCE_IDENTITY_TOL);
        assert!(
            (dist.logpdf(&case.x).unwrap() - dist_from_cov.logpdf(&case.x).unwrap()).abs()
                < FROM_COVARIANCE_IDENTITY_TOL
        );
    }

    // Test InvWishart
    for (case, resp) in invwishart_cases.iter().zip(oracle.invwishart.iter()) {
        assert_eq!(case.case_id, resp.case_id);
        let dist = InvWishart::new(case.df, &case.scale).expect("invwishart new");

        // pdf & logpdf
        let rust_pdf = dist.pdf(&case.x).expect("invwishart pdf");
        let rust_logpdf = dist.logpdf(&case.x).expect("invwishart logpdf");
        check_pair(
            &format!("{}_pdf", case.case_id),
            "InvWishart",
            rust_pdf,
            resp.pdf,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );
        check_pair(
            &format!("{}_logpdf", case.case_id),
            "InvWishart",
            rust_logpdf,
            resp.logpdf,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );

        // mean
        if let Some(scipy_m00) = resp.mean_00 {
            let rust_mean = dist.mean();
            check_pair(
                &format!("{}_mean_00", case.case_id),
                "InvWishart",
                rust_mean[0][0],
                scipy_m00,
                CLOSED_FORM_ABS_TOL,
                CLOSED_FORM_REL_TOL,
                &mut records,
            );
        }

        // c and C accessors
        let rust_chol = dist.c();
        assert_eq!(rust_chol, dist.C());
        check_pair(
            &format!("{}_chol_00", case.case_id),
            "InvWishart",
            rust_chol[0][0],
            resp.chol_00,
            CLOSED_FORM_ABS_TOL,
            CLOSED_FORM_REL_TOL,
            &mut records,
        );
        check_pair(
            &format!("{}_chol_10", case.case_id),
            "InvWishart",
            rust_chol[1][0],
            resp.chol_10,
            CLOSED_FORM_ABS_TOL,
            CLOSED_FORM_REL_TOL,
            &mut records,
        );

        // Test from_covariance parity
        let cov_rep = Covariance::from_psd(&case.scale).expect("cov psd");
        let dist_from_cov =
            InvWishart::from_covariance(case.df, &cov_rep).expect("invwishart from_cov");
        assert_eq!(dist.dim(), dist_from_cov.dim());
        assert_eq!(dist.df(), dist_from_cov.df());
        assert_eq!(dist.scale(), dist_from_cov.scale());
        assert!(
            (dist.logpdf(&case.x).unwrap() - dist_from_cov.logpdf(&case.x).unwrap()).abs()
                < FROM_COVARIANCE_IDENTITY_TOL
        );
    }

    // Test MatrixNormal
    for (case, resp) in matrix_normal_cases.iter().zip(oracle.matrix_normal.iter()) {
        assert_eq!(case.case_id, resp.case_id);
        let dist =
            MatrixNormal::new(&case.mean, &case.rowcov, &case.colcov).expect("matrix_normal new");

        assert_eq!(dist.dims(), (case.mean.len(), case.mean[0].len()));
        assert_eq!(dist.mean(), &case.mean);
        assert_eq!(dist.rowcov(), &case.rowcov);
        assert_eq!(dist.colcov(), &case.colcov);

        let rust_pdf = dist.pdf(&case.x).expect("matrix_normal pdf");
        let rust_logpdf = dist.logpdf(&case.x).expect("matrix_normal logpdf");
        check_pair(
            &format!("{}_pdf", case.case_id),
            "MatrixNormal",
            rust_pdf,
            resp.pdf,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );
        check_pair(
            &format!("{}_logpdf", case.case_id),
            "MatrixNormal",
            rust_logpdf,
            resp.logpdf,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );

        let rust_entropy = dist.entropy();
        check_pair(
            &format!("{}_entropy", case.case_id),
            "MatrixNormal",
            rust_entropy,
            resp.entropy,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );

        // from_covariance
        let cov_u = Covariance::from_psd(&case.rowcov).expect("cov_u psd");
        let cov_v = Covariance::from_psd(&case.colcov).expect("cov_v psd");
        let dist_from_cov =
            MatrixNormal::from_covariance(&case.mean, &cov_u, &cov_v).expect("from_covariance");
        assert_eq!(dist.dims(), dist_from_cov.dims());
        assert!((dist.entropy() - dist_from_cov.entropy()).abs() < FROM_COVARIANCE_IDENTITY_TOL);
        assert!(
            (dist.logpdf(&case.x).unwrap() - dist_from_cov.logpdf(&case.x).unwrap()).abs()
                < FROM_COVARIANCE_IDENTITY_TOL
        );

        // rvs differential test
        let mut rng = StdRng::seed_from_u64(42);
        let rvs_samples = dist.rvs(50, &mut rng);
        assert_eq!(rvs_samples.len(), resp.rvs_shape[0]);
        for s in &rvs_samples {
            assert_eq!(s.len(), resp.rvs_shape[1]);
            assert_eq!(s[0].len(), resp.rvs_shape[2]);
        }
        let large_samples = dist.rvs(1000, &mut rng);
        let mut sample_sum = vec![vec![0.0; case.mean[0].len()]; case.mean.len()];
        for s in &large_samples {
            for r in 0..case.mean.len() {
                for c in 0..case.mean[0].len() {
                    sample_sum[r][c] += s[r][c];
                }
            }
        }
        for r in 0..case.mean.len() {
            for c in 0..case.mean[0].len() {
                let m_est = sample_sum[r][c] / 1000.0;
                check_pair(
                    &format!("{}_rvs_mean_{r}_{c}", case.case_id),
                    "MatrixNormal",
                    m_est,
                    case.mean[r][c],
                    RVS_MEAN_TOL,
                    RVS_MEAN_TOL,
                    &mut records,
                );
            }
        }
    }

    // Test MatrixT
    for (case, resp) in matrix_t_cases.iter().zip(oracle.matrix_t.iter()) {
        assert_eq!(case.case_id, resp.case_id);
        let dist = MatrixT::new(&case.mean, &case.row_spread, &case.col_spread, case.df)
            .expect("matrix_t new");

        assert_eq!(dist.dims(), (case.mean.len(), case.mean[0].len()));
        assert_eq!(dist.mean(), &case.mean);
        assert_eq!(dist.df(), case.df);
        assert_eq!(dist.row_spread(), &case.row_spread);
        assert_eq!(dist.col_spread(), &case.col_spread);

        let rust_pdf = dist.pdf(&case.x).expect("matrix_t pdf");
        let rust_logpdf = dist.logpdf(&case.x).expect("matrix_t logpdf");
        check_pair(
            &format!("{}_pdf", case.case_id),
            "MatrixT",
            rust_pdf,
            resp.pdf,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );
        check_pair(
            &format!("{}_logpdf", case.case_id),
            "MatrixT",
            rust_logpdf,
            resp.logpdf,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );

        // from_covariance
        let cov_u = Covariance::from_psd(&case.row_spread).expect("cov_u psd");
        let cov_v = Covariance::from_psd(&case.col_spread).expect("cov_v psd");
        let dist_from_cov =
            MatrixT::from_covariance(&case.mean, &cov_u, &cov_v, case.df).expect("from_covariance");
        assert_eq!(dist.dims(), dist_from_cov.dims());
        assert_eq!(dist.df(), dist_from_cov.df());
        assert!(
            (dist.logpdf(&case.x).unwrap() - dist_from_cov.logpdf(&case.x).unwrap()).abs()
                < FROM_COVARIANCE_IDENTITY_TOL
        );

        // rvs differential test
        let mut rng = StdRng::seed_from_u64(42);
        let rvs_samples = dist.rvs(50, &mut rng).expect("matrix_t rvs");
        assert_eq!(rvs_samples.len(), resp.rvs_shape[0]);
        for s in &rvs_samples {
            assert_eq!(s.len(), resp.rvs_shape[1]);
            assert_eq!(s[0].len(), resp.rvs_shape[2]);
        }
        let large_samples = dist.rvs(1000, &mut rng).expect("matrix_t rvs large");
        let mut sample_sum = vec![vec![0.0; case.mean[0].len()]; case.mean.len()];
        for s in &large_samples {
            for r in 0..case.mean.len() {
                for c in 0..case.mean[0].len() {
                    sample_sum[r][c] += s[r][c];
                }
            }
        }
        for r in 0..case.mean.len() {
            for c in 0..case.mean[0].len() {
                let m_est = sample_sum[r][c] / 1000.0;
                check_pair(
                    &format!("{}_rvs_mean_{r}_{c}", case.case_id),
                    "MatrixT",
                    m_est,
                    case.mean[r][c],
                    MATRIX_T_RVS_MEAN_TOL,
                    MATRIX_T_RVS_MEAN_TOL,
                    &mut records,
                );
            }
        }
    }

    // Test VonMisesFisher
    for (case, resp) in vmf_cases.iter().zip(oracle.vmf.iter()) {
        assert_eq!(case.case_id, resp.case_id);
        let dist = VonMisesFisher::new(&case.mu, case.kappa);

        assert_eq!(dist.dim(), case.mu.len());
        assert_eq!(dist.mu(), &case.mu);
        assert_eq!(dist.kappa(), case.kappa);

        let rust_pdf = dist.pdf(&case.x);
        let rust_logpdf = dist.logpdf(&case.x);
        check_pair(
            &format!("{}_pdf", case.case_id),
            "VonMisesFisher",
            rust_pdf,
            resp.pdf,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );
        check_pair(
            &format!("{}_logpdf", case.case_id),
            "VonMisesFisher",
            rust_logpdf,
            resp.logpdf,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );

        let rust_entropy = dist.entropy();
        check_pair(
            &format!("{}_entropy", case.case_id),
            "VonMisesFisher",
            rust_entropy,
            resp.entropy,
            SERIES_ENTROPY_ABS_TOL,
            SERIES_ENTROPY_REL_TOL,
            &mut records,
        );

        // rvs differential test
        let mut rng = StdRng::seed_from_u64(42);
        let rvs_samples = dist.rvs(50, &mut rng);
        assert_eq!(rvs_samples.len(), resp.rvs_shape[0]);
        let mut max_norm_err = 0.0_f64;
        let mut mean_dir = vec![0.0; case.mu.len()];
        for s in &rvs_samples {
            assert_eq!(s.len(), resp.rvs_shape[1]);
            let norm: f64 = s.iter().map(|&v| v * v).sum::<f64>().sqrt();
            max_norm_err = max_norm_err.max((norm - 1.0).abs());
            for (idx, &v) in s.iter().enumerate() {
                mean_dir[idx] += v;
            }
        }
        check_pair(
            &format!("{}_rvs_unit_norm_err", case.case_id),
            "VonMisesFisher",
            max_norm_err,
            resp.rvs_norm_err,
            UNIT_NORM_ERR_TOL,
            UNIT_NORM_ERR_TOL,
            &mut records,
        );
        let dot_mu: f64 = mean_dir.iter().zip(&case.mu).map(|(&d, &m)| d * m).sum();
        assert!(dot_mu > 0.0, "vmf sample mean aligns with concentration mu");
    }

    // Test Dirichlet
    for (case, resp) in dirichlet_cases.iter().zip(oracle.dirichlet.iter()) {
        assert_eq!(case.case_id, resp.case_id);
        let dist = Dirichlet::new(&case.alpha);

        assert_eq!(dist.dim(), case.alpha.len());

        let rust_pdf = dist.pdf(&case.x);
        let rust_logpdf = dist.logpdf(&case.x);
        check_pair(
            &format!("{}_pdf", case.case_id),
            "Dirichlet",
            rust_pdf,
            resp.pdf,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );
        check_pair(
            &format!("{}_logpdf", case.case_id),
            "Dirichlet",
            rust_logpdf,
            resp.logpdf,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );

        let rust_mean = dist.mean();
        for (k, (&rm, &sm)) in rust_mean.iter().zip(&resp.mean).enumerate() {
            check_pair(
                &format!("{}_mean_{k}", case.case_id),
                "Dirichlet",
                rm,
                sm,
                CLOSED_FORM_ABS_TOL,
                CLOSED_FORM_REL_TOL,
                &mut records,
            );
        }

        let rust_var = dist.var();
        for (k, (&rv, &sv)) in rust_var.iter().zip(&resp.var).enumerate() {
            check_pair(
                &format!("{}_var_{k}", case.case_id),
                "Dirichlet",
                rv,
                sv,
                CLOSED_FORM_ABS_TOL,
                CLOSED_FORM_REL_TOL,
                &mut records,
            );
        }

        let rust_cov = dist.cov();
        for i in 0..case.alpha.len() {
            for j in 0..case.alpha.len() {
                check_pair(
                    &format!("{}_cov_{i}_{j}", case.case_id),
                    "Dirichlet",
                    rust_cov[i][j],
                    resp.cov[i][j],
                    CLOSED_FORM_ABS_TOL,
                    CLOSED_FORM_REL_TOL,
                    &mut records,
                );
            }
        }
    }

    // Test MultivariateHypergeom
    for (case, resp) in mhypergeom_cases.iter().zip(oracle.mhypergeom.iter()) {
        assert_eq!(case.case_id, resp.case_id);
        let dist = MultivariateHypergeom::new(&case.m, case.n);

        assert_eq!(dist.dim(), case.m.len());
        assert_eq!(dist.m(), &case.m);
        assert_eq!(dist.n(), case.n);
        assert_eq!(dist.total(), case.m.iter().sum::<usize>());

        let rust_pmf = dist.pmf(&case.x);
        let rust_logpmf = dist.logpmf(&case.x);
        check_pair(
            &format!("{}_pmf", case.case_id),
            "MultivariateHypergeom",
            rust_pmf,
            resp.pmf,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );
        check_pair(
            &format!("{}_logpmf", case.case_id),
            "MultivariateHypergeom",
            rust_logpmf,
            resp.logpmf,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );

        let rust_mean = dist.mean();
        for (k, (&rm, &sm)) in rust_mean.iter().zip(&resp.mean).enumerate() {
            check_pair(
                &format!("{}_mean_{k}", case.case_id),
                "MultivariateHypergeom",
                rm,
                sm,
                CLOSED_FORM_ABS_TOL,
                CLOSED_FORM_REL_TOL,
                &mut records,
            );
        }

        let rust_var = dist.var();
        for (k, (&rv, &sv)) in rust_var.iter().zip(&resp.var).enumerate() {
            check_pair(
                &format!("{}_var_{k}", case.case_id),
                "MultivariateHypergeom",
                rv,
                sv,
                CLOSED_FORM_ABS_TOL,
                CLOSED_FORM_REL_TOL,
                &mut records,
            );
        }

        let rust_cov = dist.cov();
        for i in 0..case.m.len() {
            for j in 0..case.m.len() {
                check_pair(
                    &format!("{}_cov_{i}_{j}", case.case_id),
                    "MultivariateHypergeom",
                    rust_cov[i][j],
                    resp.cov[i][j],
                    CLOSED_FORM_ABS_TOL,
                    CLOSED_FORM_REL_TOL,
                    &mut records,
                );
            }
        }

        // rvs differential test
        let mut rng = StdRng::seed_from_u64(42);
        let rvs_samples = dist.rvs(50, &mut rng);
        assert_eq!(rvs_samples.len(), resp.rvs_shape[0]);
        for s in &rvs_samples {
            assert_eq!(s.len(), resp.rvs_shape[1]);
            assert_eq!(s.iter().sum::<usize>(), case.n);
            for (idx, &v) in s.iter().enumerate() {
                assert!(v <= case.m[idx]);
            }
        }
        let large_samples = dist.rvs(1000, &mut rng);
        let mut sample_sum = vec![0.0; case.m.len()];
        for s in &large_samples {
            for (idx, &v) in s.iter().enumerate() {
                sample_sum[idx] += v as f64;
            }
        }
        for (k, (&sm, &exp_m)) in sample_sum.iter().zip(&resp.mean).enumerate() {
            let m_est = sm / 1000.0;
            check_pair(
                &format!("{}_rvs_mean_{k}", case.case_id),
                "MultivariateHypergeom",
                m_est,
                exp_m,
                MVHYPERGEOM_RVS_MEAN_TOL,
                MVHYPERGEOM_RVS_MEAN_TOL,
                &mut records,
            );
        }
    }

    // Test NormalInverseGamma
    for (case, resp) in nig_cases.iter().zip(oracle.nig.iter()) {
        assert_eq!(case.case_id, resp.case_id);
        let dist = NormalInverseGamma::new(case.mu, case.lmbda, case.a, case.b);

        assert_eq!(dist.mu(), case.mu);
        assert_eq!(dist.lmbda(), case.lmbda);
        assert_eq!(dist.a(), case.a);
        assert_eq!(dist.b(), case.b);

        let rust_pdf = dist.pdf(case.x, case.s2);
        let rust_logpdf = dist.logpdf(case.x, case.s2);
        check_pair(
            &format!("{}_pdf", case.case_id),
            "NormalInverseGamma",
            rust_pdf,
            resp.pdf,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );
        check_pair(
            &format!("{}_logpdf", case.case_id),
            "NormalInverseGamma",
            rust_logpdf,
            resp.logpdf,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );

        let (mean_x, mean_s2) = dist.mean();
        check_pair(
            &format!("{}_mean_x", case.case_id),
            "NormalInverseGamma",
            mean_x,
            resp.mean_x,
            CLOSED_FORM_ABS_TOL,
            CLOSED_FORM_REL_TOL,
            &mut records,
        );
        check_pair(
            &format!("{}_mean_s2", case.case_id),
            "NormalInverseGamma",
            mean_s2,
            resp.mean_s2,
            CLOSED_FORM_ABS_TOL,
            CLOSED_FORM_REL_TOL,
            &mut records,
        );

        let (var_x, var_s2) = dist.var();
        check_pair(
            &format!("{}_var_x", case.case_id),
            "NormalInverseGamma",
            var_x,
            resp.var_x,
            CLOSED_FORM_ABS_TOL,
            CLOSED_FORM_REL_TOL,
            &mut records,
        );
        check_pair(
            &format!("{}_var_s2", case.case_id),
            "NormalInverseGamma",
            var_s2,
            resp.var_s2,
            CLOSED_FORM_ABS_TOL,
            CLOSED_FORM_REL_TOL,
            &mut records,
        );

        // rvs differential test
        let mut rng = StdRng::seed_from_u64(42);
        let rvs_samples = dist.rvs(50, &mut rng);
        assert_eq!(rvs_samples.len(), resp.rvs_len);
        for &(x_val, s2_val) in &rvs_samples {
            assert!(s2_val > 0.0, "s2 must be strictly positive");
            assert!(x_val.is_finite());
        }
        let large_samples = dist.rvs(2000, &mut rng);
        let mut sum_x = 0.0;
        let mut sum_s2 = 0.0;
        for &(x_val, s2_val) in &large_samples {
            sum_x += x_val;
            sum_s2 += s2_val;
        }
        let est_mean_x = sum_x / 2000.0;
        let est_mean_s2 = sum_s2 / 2000.0;
        check_pair(
            &format!("{}_rvs_mean_x", case.case_id),
            "NormalInverseGamma",
            est_mean_x,
            resp.mean_x,
            RVS_MEAN_TOL,
            RVS_MEAN_TOL,
            &mut records,
        );
        check_pair(
            &format!("{}_rvs_mean_s2", case.case_id),
            "NormalInverseGamma",
            est_mean_s2,
            resp.mean_s2,
            RVS_MEAN_TOL,
            RVS_MEAN_TOL,
            &mut records,
        );
    }

    // Test Multinomial
    for (case, resp) in multinomial_cases.iter().zip(oracle.multinomial.iter()) {
        assert_eq!(case.case_id, resp.case_id);
        let dist = Multinomial::new(case.n, &case.p);

        assert_eq!(dist.dim(), case.p.len());
        assert_eq!(dist.n(), case.n);
        assert_eq!(dist.p(), &case.p);

        let rust_pmf = dist.pmf(&case.x);
        let rust_logpmf = dist.logpmf(&case.x);
        check_pair(
            &format!("{}_pmf", case.case_id),
            "Multinomial",
            rust_pmf,
            resp.pmf,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );
        check_pair(
            &format!("{}_logpmf", case.case_id),
            "Multinomial",
            rust_logpmf,
            resp.logpmf,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );

        let rust_mean = dist.mean();
        for (k, (&rm, &sm)) in rust_mean.iter().zip(&resp.mean).enumerate() {
            check_pair(
                &format!("{}_mean_{k}", case.case_id),
                "Multinomial",
                rm,
                sm,
                CLOSED_FORM_ABS_TOL,
                CLOSED_FORM_REL_TOL,
                &mut records,
            );
        }

        let rust_cov = dist.cov();
        for i in 0..case.p.len() {
            for j in 0..case.p.len() {
                check_pair(
                    &format!("{}_cov_{i}_{j}", case.case_id),
                    "Multinomial",
                    rust_cov[i][j],
                    resp.cov[i][j],
                    CLOSED_FORM_ABS_TOL,
                    CLOSED_FORM_REL_TOL,
                    &mut records,
                );
            }
        }

        let rust_var = dist.var();
        for (k, &rv) in rust_var.iter().enumerate() {
            check_pair(
                &format!("{}_var_{k}", case.case_id),
                "Multinomial",
                rv,
                resp.cov[k][k],
                CLOSED_FORM_ABS_TOL,
                CLOSED_FORM_REL_TOL,
                &mut records,
            );
        }

        let rust_entropy = dist.entropy();
        check_pair(
            &format!("{}_entropy", case.case_id),
            "Multinomial",
            rust_entropy,
            resp.entropy,
            SERIES_ENTROPY_ABS_TOL,
            SERIES_ENTROPY_REL_TOL,
            &mut records,
        );

        // rvs differential test
        let mut rng = StdRng::seed_from_u64(42);
        let rvs_samples = dist.rvs(50, &mut rng);
        assert_eq!(rvs_samples.len(), resp.rvs_shape[0]);
        for s in &rvs_samples {
            assert_eq!(s.len(), resp.rvs_shape[1]);
            assert_eq!(s.iter().sum::<usize>(), case.n);
        }
        let large_samples = dist.rvs(1500, &mut rng);
        let mut sample_sum = vec![0.0; case.p.len()];
        for s in &large_samples {
            for (idx, &v) in s.iter().enumerate() {
                sample_sum[idx] += v as f64;
            }
        }
        for (k, (&sm, &exp_m)) in sample_sum.iter().zip(&resp.mean).enumerate() {
            let m_est = sm / 1500.0;
            check_pair(
                &format!("{}_rvs_mean_{k}", case.case_id),
                "Multinomial",
                m_est,
                exp_m,
                RVS_MEAN_TOL,
                RVS_MEAN_TOL,
                &mut records,
            );
        }
    }

    // Test DirichletMultinomial
    for (case, resp) in dirichlet_multinomial_cases
        .iter()
        .zip(oracle.dirichlet_multinomial.iter())
    {
        assert_eq!(case.case_id, resp.case_id);
        let dist = DirichletMultinomial::new(&case.alpha, case.n);

        assert_eq!(dist.dim(), case.alpha.len());
        assert_eq!(dist.n(), case.n);
        assert_eq!(dist.alpha(), &case.alpha);

        let rust_pmf = dist.pmf(&case.x);
        let rust_logpmf = dist.logpmf(&case.x);
        check_pair(
            &format!("{}_pmf", case.case_id),
            "DirichletMultinomial",
            rust_pmf,
            resp.pmf,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );
        check_pair(
            &format!("{}_logpmf", case.case_id),
            "DirichletMultinomial",
            rust_logpmf,
            resp.logpmf,
            SPECIAL_FN_ABS_TOL,
            SPECIAL_FN_REL_TOL,
            &mut records,
        );

        let rust_mean = dist.mean();
        for (k, (&rm, &sm)) in rust_mean.iter().zip(&resp.mean).enumerate() {
            check_pair(
                &format!("{}_mean_{k}", case.case_id),
                "DirichletMultinomial",
                rm,
                sm,
                CLOSED_FORM_ABS_TOL,
                CLOSED_FORM_REL_TOL,
                &mut records,
            );
        }

        let rust_var = dist.var();
        for (k, (&rv, &sv)) in rust_var.iter().zip(&resp.var).enumerate() {
            check_pair(
                &format!("{}_var_{k}", case.case_id),
                "DirichletMultinomial",
                rv,
                sv,
                CLOSED_FORM_ABS_TOL,
                CLOSED_FORM_REL_TOL,
                &mut records,
            );
        }

        let rust_cov = dist.cov();
        for i in 0..case.alpha.len() {
            for j in 0..case.alpha.len() {
                check_pair(
                    &format!("{}_cov_{i}_{j}", case.case_id),
                    "DirichletMultinomial",
                    rust_cov[i][j],
                    resp.cov[i][j],
                    CLOSED_FORM_ABS_TOL,
                    CLOSED_FORM_REL_TOL,
                    &mut records,
                );
            }
        }
    }

    // The random matrix / direction / table generators are compared distributionally in
    // `diff_random_generators_match_scipys_distributions` below (frankenscipy-olv0j.9).

    let duration_ns = t0.elapsed().as_nanos();
    let max_abs_diff = records.iter().map(|r| r.abs_diff).fold(0.0_f64, f64::max);
    let max_rel_diff = records.iter().map(|r| r.rel_diff).fold(0.0_f64, f64::max);
    let all_pass = records.iter().all(|r| r.pass);

    let log = DiffLog {
        test_id: "diff_stats_multivariate".into(),
        category: "stats.multivariate_distributions".into(),
        case_count: records.len(),
        max_abs_diff,
        max_rel_diff,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns,
        records,
    };
    emit_log(&log);
    assert!(
        all_pass,
        "all multivariate differential test cases must pass"
    );
}

// ── Random generators, compared by distribution (frankenscipy-olv0j.9) ──────────────────
//
// These generators draw from different RNG streams than SciPy's, so no single sample can
// be compared. They used to be "compared" by checks the identity matrix passes (orthogonal,
// unit norm, unit diagonal) and by a literal 1.0 held against SciPy's |det|; the stubs that
// returned I, a zero table and a constant survived them for a week. Here each generator's
// marginals are held against a large SciPy sample with a two-sample Kolmogorov-Smirnov
// test, the invariants are checked on fsci's OWN samples, and every detector is shown to
// reject the old stub's output (the must-miss arm).

/// Draws per generator per dimension, on each side.
const GENERATOR_DRAWS: usize = 4000;
/// Two-sample KS rejection level per comparison. There are about 26 statistical checks per
/// run, so a correct generator fails a run with probability below 1e-6.
const KS_ALPHA: f64 = 2.5e-8;
/// |z| bound for the det-sign balance and the mean tests (two-sided p about 4e-8).
const Z_BOUND: f64 = 5.5;
/// A correlation matrix with the requested spectrum has ||R||_F^2 = sum(eig^2), relative.
const SPECTRUM_FROBENIUS_TOL: f64 = 1e-9;
/// |R[0,1]| below this is a structural zero's rounding residue (see the random_correlation
/// comparison). In 20000 SciPy draws the residues run from 1e-19 to 1.7e-12 and every other
/// value is at least 1e-3, with nothing in between, so the cut sits inside that gap.
const STRUCTURAL_ZERO_SNAP: f64 = 1e-9;

#[derive(Debug, Deserialize)]
struct GeneratorSamples {
    q00: Vec<f64>,
    so00: Vec<f64>,
    u00_abs2: Vec<f64>,
    u00_arg: Vec<f64>,
    v0: Vec<f64>,
    r01: Vec<f64>,
    t00: Vec<f64>,
}

fn scipy_generator_samples(
    dims: &[usize],
    eigs: &[Vec<f64>],
) -> Option<std::collections::BTreeMap<String, GeneratorSamples>> {
    let script = r#"
import json, sys
import numpy as np
from scipy import stats
q = json.load(sys.stdin)
n = q["draws"]
out = {}
for dim, eigs in zip(q["dims"], q["eigs"]):
    rng = np.random.default_rng(20260925 + dim)
    o = stats.ortho_group.rvs(dim, size=n, random_state=rng)
    so = stats.special_ortho_group.rvs(dim, size=n, random_state=rng)
    u = stats.unitary_group.rvs(dim, size=n, random_state=rng)
    v = stats.uniform_direction.rvs(dim, size=n, random_state=rng)
    eigs = np.array(eigs)
    r01 = [float(stats.random_correlation.rvs(eigs, random_state=rng)[0, 1]) for _ in range(n)]
    t = stats.random_table.rvs([10 * dim, 20 * dim], [15 * dim, 15 * dim], size=n,
                               random_state=rng)
    out[str(dim)] = {
        "q00": o[:, 0, 0].tolist(),
        "so00": so[:, 0, 0].tolist(),
        "u00_abs2": (np.abs(u[:, 0, 0]) ** 2).tolist(),
        "u00_arg": np.angle(u[:, 0, 0]).tolist(),
        "v0": v[:, 0].tolist(),
        "r01": r01,
        "t00": t[:, 0, 0].astype(float).tolist(),
    }
json.dump(out, sys.stdout)
"#;
    let query = serde_json::json!({ "draws": GENERATOR_DRAWS, "dims": dims, "eigs": eigs });
    let mut child = match fsci_conformance::scipy_oracle_command()
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn the generator oracle: {e}"
            );
            return None;
        }
    };
    child
        .stdin
        .as_mut()
        .expect("oracle stdin")
        .write_all(query.to_string().as_bytes())
        .expect("write generator query");
    let output = child
        .wait_with_output()
        .expect("wait for the generator oracle");
    if !output.status.success() {
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "generator oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("parse generator oracle JSON"))
}

/// Two-sample Kolmogorov-Smirnov statistic, correct under ties (both ECDFs are advanced past
/// every copy of a value before they are compared).
fn ks_statistic(a: &[f64], b: &[f64]) -> f64 {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    a.sort_by(f64::total_cmp);
    b.sort_by(f64::total_cmp);
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let (mut i, mut j, mut d) = (0usize, 0usize, 0.0_f64);
    while i < a.len() && j < b.len() {
        let x = a[i].min(b[j]);
        while i < a.len() && a[i] <= x {
            i += 1;
        }
        while j < b.len() && b[j] <= x {
            j += 1;
        }
        d = d.max((i as f64 / na - j as f64 / nb).abs());
    }
    d
}

/// The asymptotic two-sample KS critical value at `KS_ALPHA`.
fn ks_critical(na: usize, nb: usize) -> f64 {
    let (na, nb) = (na as f64, nb as f64);
    (-(KS_ALPHA / 2.0).ln() / 2.0).sqrt() * ((na + nb) / (na * nb)).sqrt()
}

/// The 10th, 50th and 90th percentiles, rounded, for the log.
fn deciles(v: &[f64]) -> [f64; 3] {
    if v.is_empty() {
        return [f64::NAN; 3];
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let at = |q: f64| {
        let x = s[((s.len() - 1) as f64 * q).round() as usize];
        (x * 1e4).round() / 1e4
    };
    [at(0.1), at(0.5), at(0.9)]
}

fn ks_rejects(a: &[f64], b: &[f64]) -> bool {
    ks_statistic(a, b) > ks_critical(a.len(), b.len())
}

/// Determinant by Gaussian elimination with partial pivoting.
fn det(m: &[Vec<f64>]) -> f64 {
    let n = m.len();
    let mut a: Vec<Vec<f64>> = m.to_vec();
    let mut product = 1.0;
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))
            .expect("non-empty column");
        if a[pivot][col] == 0.0 {
            return 0.0;
        }
        if pivot != col {
            a.swap(pivot, col);
            product = -product;
        }
        product *= a[col][col];
        for row in col + 1..n {
            let factor = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
        }
    }
    product
}

fn max_orthonormality_error(q: &[Vec<f64>]) -> f64 {
    let n = q.len();
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let dot: f64 = (0..n).map(|k| q[i][k] * q[j][k]).sum();
            worst = worst.max((dot - f64::from(u8::from(i == j))).abs());
        }
    }
    worst
}

fn max_unitarity_error(u: &[Vec<(f64, f64)>]) -> f64 {
    let n = u.len();
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let (mut re, mut im) = (0.0, 0.0);
            for k in 0..n {
                let (a_re, a_im) = u[i][k];
                let (b_re, b_im) = u[j][k];
                re += a_re * b_re + a_im * b_im;
                im += a_im * b_re - a_re * b_im;
            }
            worst = worst.max((re - f64::from(u8::from(i == j))).abs().max(im.abs()));
        }
    }
    worst
}

/// |z| of `successes` out of `n` Bernoulli(1/2) trials.
fn sign_balance_z(successes: usize, n: usize) -> f64 {
    (successes as f64 - n as f64 / 2.0).abs() / (n as f64 / 4.0).sqrt()
}

fn margins_are(table: &[Vec<usize>], rows: &[usize], cols: &[usize]) -> bool {
    table.len() == rows.len()
        && table
            .iter()
            .zip(rows)
            .all(|(row, &r)| row.len() == cols.len() && row.iter().sum::<usize>() == r)
        && (0..cols.len()).all(|c| table.iter().map(|row| row[c]).sum::<usize>() == cols[c])
}

fn frobenius_matches_spectrum(r: &[Vec<f64>], eigs: &[f64]) -> bool {
    let frob: f64 = r.iter().flatten().map(|v| v * v).sum();
    let spectrum: f64 = eigs.iter().map(|e| e * e).sum();
    (frob - spectrum).abs() <= SPECTRUM_FROBENIUS_TOL * spectrum
}

#[test]
fn diff_random_generators_match_scipys_distributions() {
    let dims = [3usize, 5];
    // SciPy's own example spectrum shape: linspace(1.5, 0.5) rescaled to sum to dim.
    let eigs: Vec<Vec<f64>> = dims
        .iter()
        .map(|&n| {
            let raw: Vec<f64> = (0..n).map(|i| 1.5 - i as f64 / (n as f64 - 1.0)).collect();
            let sum: f64 = raw.iter().sum();
            raw.iter().map(|v| v * n as f64 / sum).collect()
        })
        .collect();
    let Some(scipy) = scipy_generator_samples(&dims, &eigs) else {
        return;
    };
    let draws = GENERATOR_DRAWS;
    let mut failures = Vec::new();
    let mut ks_checks = 0usize;
    let mut ks = |name: String, fsci: &[f64], scipy: &[f64], failures: &mut Vec<String>| {
        ks_checks += 1;
        // A generator that failed its invariant stops early; a short sample must not pass
        // (the KS statistic of an empty sample is 0).
        if fsci.len() != draws || scipy.len() != draws {
            failures.push(format!(
                "{name}: compared {} fsci and {} SciPy draws, not {draws}",
                fsci.len(),
                scipy.len()
            ));
            return;
        }
        let (d, crit) = (
            ks_statistic(fsci, scipy),
            ks_critical(fsci.len(), scipy.len()),
        );
        println!(
            "{name}: KS D = {d:.4} (critical {crit:.4}); deciles 1/5/9 fsci {:?} scipy {:?}",
            deciles(fsci),
            deciles(scipy)
        );
        if d > crit {
            failures.push(format!("{name}: KS D = {d:.4} > {crit:.4}"));
        }
    };

    for (&n, eigs) in dims.iter().zip(&eigs) {
        let s = &scipy[&n.to_string()];
        let mut rng = StdRng::seed_from_u64(20_260_925 + n as u64);

        // ortho_group: Haar on O(n). det is +1 or -1 with probability 1/2 each.
        let mut q00 = Vec::with_capacity(draws);
        let mut positive = 0usize;
        for _ in 0..draws {
            let q = ortho_group::rvs_with_rng(n, &mut rng);
            let d = det(&q);
            if max_orthonormality_error(&q) > ORTHONORMALITY_TOL
                || (d.abs() - 1.0).abs() > ORTHO_DET_TOL
            {
                failures.push(format!("ortho_group n={n}: not orthogonal (det {d})"));
                break;
            }
            positive += usize::from(d > 0.0);
            q00.push(q[0][0]);
        }
        let z = sign_balance_z(positive, draws);
        if z > Z_BOUND {
            failures.push(format!(
                "ortho_group n={n}: det > 0 in {positive} of {draws} (z = {z:.2})"
            ));
        }
        ks(
            format!("ortho_group n={n} Q[0,0]"),
            &q00,
            &s.q00,
            &mut failures,
        );

        // special_ortho_group: Haar on SO(n); every det is +1.
        let mut so00 = Vec::with_capacity(draws);
        for _ in 0..draws {
            let q = special_ortho_group::rvs_with_rng(n, &mut rng);
            let d = det(&q);
            if max_orthonormality_error(&q) > ORTHONORMALITY_TOL || (d - 1.0).abs() > ORTHO_DET_TOL
            {
                failures.push(format!("special_ortho_group n={n}: det {d}"));
                break;
            }
            so00.push(q[0][0]);
        }
        ks(
            format!("special_ortho_group n={n} Q[0,0]"),
            &so00,
            &s.so00,
            &mut failures,
        );

        // unitary_group: Haar on U(n).
        let (mut abs2, mut arg) = (Vec::with_capacity(draws), Vec::with_capacity(draws));
        for _ in 0..draws {
            let u = unitary_group::rvs_with_rng(n, &mut rng);
            if max_unitarity_error(&u) > ORTHONORMALITY_TOL {
                failures.push(format!("unitary_group n={n}: not unitary"));
                break;
            }
            let (re, im) = u[0][0];
            abs2.push(re * re + im * im);
            arg.push(im.atan2(re));
        }
        ks(
            format!("unitary_group n={n} |U[0,0]|^2"),
            &abs2,
            &s.u00_abs2,
            &mut failures,
        );
        ks(
            format!("unitary_group n={n} arg U[0,0]"),
            &arg,
            &s.u00_arg,
            &mut failures,
        );

        // uniform_direction: uniform on the unit sphere; each coordinate has mean 0 and
        // variance 1/n.
        let mut v0 = Vec::with_capacity(draws);
        let mut sums = vec![0.0_f64; n];
        for _ in 0..draws {
            let v = uniform_direction::rvs_with_rng(n, &mut rng);
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if (norm - 1.0).abs() > ORTHONORMALITY_TOL {
                failures.push(format!("uniform_direction n={n}: norm {norm}"));
                break;
            }
            for (acc, x) in sums.iter_mut().zip(&v) {
                *acc += x;
            }
            v0.push(v[0]);
        }
        let se = (1.0 / n as f64 / draws as f64).sqrt();
        for (i, sum) in sums.iter().enumerate() {
            let z = (sum / draws as f64).abs() / se;
            if z > Z_BOUND {
                failures.push(format!(
                    "uniform_direction n={n}: coordinate {i} mean z = {z:.2}"
                ));
            }
        }
        ks(
            format!("uniform_direction n={n} v[0]"),
            &v0,
            &s.v0,
            &mut failures,
        );

        // random_correlation: unit diagonal and the requested spectrum, varying draw to draw.
        let mut r01 = Vec::with_capacity(draws);
        for _ in 0..draws {
            let r = random_correlation::rvs_with_rng(eigs, &mut rng);
            let diag_ok = (0..n).all(|i| (r[i][i] - 1.0).abs() <= CORRELATION_DIAG_TOL);
            if !diag_ok || !frobenius_matches_spectrum(&r, eigs) {
                failures.push(format!(
                    "random_correlation n={n}: not a correlation matrix with spectrum {eigs:?}"
                ));
                break;
            }
            r01.push(r[0][1]);
        }
        let mut distinct = r01.clone();
        distinct.sort_by(f64::total_cmp);
        distinct.dedup();
        if distinct.len() < draws / 2 {
            failures.push(format!(
                "random_correlation n={n}: only {} distinct R[0,1] in {draws} draws",
                distinct.len()
            ));
        }
        // For n = 3 with this spectrum, R[0,1] is structurally zero in about 59% of draws
        // (SciPy: 58.6% of 20000 below 1e-11), and the sign of the rounding residue there
        // is an artifact of each library's arithmetic (SciPy's splits 29.4% / 29.1%).
        // On raw values KS compares those residue signs, so both samples are snapped to 0
        // below STRUCTURAL_ZERO_SNAP first: the atom's MASS is still compared, and so is
        // everything off it.
        let snap = |v: &[f64]| -> Vec<f64> {
            v.iter()
                .map(|&x| {
                    if x.abs() < STRUCTURAL_ZERO_SNAP {
                        0.0
                    } else {
                        x
                    }
                })
                .collect()
        };
        let (fsci_r01, scipy_r01) = (snap(&r01), snap(&s.r01));
        let atom = |v: &[f64]| v.iter().filter(|&&x| x == 0.0).count() as f64 / v.len() as f64;
        println!(
            "random_correlation n={n}: share of R[0,1] at the structural zero, fsci {:.4} scipy {:.4}",
            atom(&fsci_r01),
            atom(&scipy_r01)
        );
        ks(
            format!("random_correlation n={n} R[0,1]"),
            &fsci_r01,
            &scipy_r01,
            &mut failures,
        );

        // random_table: exact margins; T[0,0] is hypergeometric.
        let rows = [10 * n, 20 * n];
        let cols = [15 * n, 15 * n];
        let mut t00 = Vec::with_capacity(draws);
        for _ in 0..draws {
            let t = random_table::rvs_with_rng(&rows, &cols, &mut rng);
            if !margins_are(&t, &rows, &cols) {
                failures.push(format!("random_table n={n}: margins {t:?}"));
                break;
            }
            t00.push(t[0][0] as f64);
        }
        let total = (rows[0] + rows[1]) as f64;
        let (r0, r1, c0, c1) = (
            rows[0] as f64,
            rows[1] as f64,
            cols[0] as f64,
            cols[1] as f64,
        );
        let mean = r0 * c0 / total;
        let var = r0 * r1 * c0 * c1 / (total * total * (total - 1.0));
        let z = (t00.iter().sum::<f64>() / draws as f64 - mean).abs() / (var / draws as f64).sqrt();
        if z > Z_BOUND {
            failures.push(format!("random_table n={n}: T[0,0] mean z = {z:.2}"));
        }
        ks(
            format!("random_table n={n} T[0,0]"),
            &t00,
            &s.t00,
            &mut failures,
        );

        // Must-miss: each detector rejects the stub output these generators used to return
        // (identity matrices, the first basis vector, the identity correlation matrix, a zero
        // table) and, for O(5), a plausible wrong marginal (uniform on [-1, 1], which is the
        // right marginal only for n = 3).
        let ones = vec![1.0; draws];
        let zeros = vec![0.0; draws];
        let stubs_seen = ks_rejects(&ones, &s.q00)
            && ks_rejects(&ones, &s.so00)
            && ks_rejects(&ones, &s.u00_abs2)
            && ks_rejects(&zeros, &s.u00_arg)
            && ks_rejects(&ones, &s.v0)
            && ks_rejects(&zeros, &s.r01)
            && ks_rejects(&zeros, &s.t00)
            && sign_balance_z(draws, draws) > Z_BOUND
            && !margins_are(&[vec![0, 0], vec![0, 0]], &rows, &cols)
            && !frobenius_matches_spectrum(&identity_rows(n), eigs);
        assert!(stubs_seen, "n={n}: a detector accepted the old stub output");
        if n == 5 {
            let uniform: Vec<f64> = (0..draws)
                .map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / draws as f64)
                .collect();
            assert!(
                ks_rejects(&uniform, &s.q00),
                "the O(5) detector accepted uniform[-1, 1] as Haar's Q[0,0] marginal"
            );
        }
    }

    println!("{ks_checks} KS comparisons against SciPy samples of {draws}");
    assert_eq!(ks_checks, 7 * dims.len(), "a KS comparison was skipped");
    assert!(
        failures.is_empty(),
        "random generators disagree with SciPy's distributions:\n{}",
        failures.join("\n")
    );
}

fn identity_rows(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| f64::from(u8::from(i == j))).collect())
        .collect()
}
