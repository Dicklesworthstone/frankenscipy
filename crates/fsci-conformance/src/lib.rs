#![forbid(unsafe_code)]

//! SciPy-parity conformance harness.
//!
//! # Three entry-point hierarchies (per frankenscipy-aqm6)
//!
//! The crate exposes THREE parallel ways to run conformance for a family:
//!
//! 1. **`run_<family>_packet(config, fixture_name)`** — self-check lane.
//!    Executes the Rust implementation against each case in the fixture
//!    and compares the result to the fixture-embedded `expected` value.
//!    NO scipy oracle is consulted. Use when you want a pure
//!    reproducibility check of the Rust side. Families:
//!    `validate_tol`, `linalg`, `optimize`, `special`, `array_api`,
//!    `sparse`, `fft`, `casp`, `cluster`, `spatial`, `signal`, `stats`,
//!    `integrate`, `interpolate`, `ndimage`, `io`.
//!
//! 2. **`run_<family>_packet_with_oracle_capture(config, fixture_name, oracle)`**
//!    — oracle-backed lane. Invokes the scipy Python oracle to capture
//!    reference outputs, then compares the Rust implementation against
//!    those. Only `linalg` has this variant today
//!    (`run_linalg_packet_with_oracle_capture`). New oracle-backed
//!    entry points should follow this naming convention.
//!
//! 3. **`run_differential_test(fixture_path, oracle_config)`** — dispatch
//!    front door. Reads the fixture envelope, routes to
//!    `run_differential_<family>` based on `family` (exact-match per
//!    frankenscipy-ubkg), and emits per-case differential artifacts
//!    with audit ledgers for all 12 families per frankenscipy-mhuk.
//!    Most runners today fall back to self-check semantics because
//!    their scipy oracle is absent or wiring is incomplete; tracked by
//!    frankenscipy-ivg5.
//!
//! ## Which to call?
//!
//! - **Tests and quick local runs**: `run_<family>_packet`.
//! - **Oracle-backed parity verification**: `run_<family>_packet_with_oracle_capture`
//!   where available, or `run_differential_test` for the dispatch path.
//! - **CI G3/G5**: `run_differential_test` aggregates dispatch and
//!   artifact emission.
//!
//! ## Report provenance
//!
//! Each `PacketReport` carries a `report_kind` tag
//! (`Unspecified`/`OracleBacked`/`SelfCheck`, per frankenscipy-fytm) so
//! downstream dashboards (crate `dashboard`) can distinguish the lanes.

pub mod ci_gates;
pub mod dashboard;
pub mod e2e;
pub mod forensics;
pub mod metamorphic;
pub mod quality_gates;

use asupersync::raptorq::decoder::{InactivationDecoder, ReceivedSymbol};
use asupersync::raptorq::systematic::SystematicEncoder;
use blake3::hash;
use fsci_arrayapi::{
    ArangeRequest as ArrayApiArangeRequest, ArrayApiArray as FsciArrayApiArray,
    CoreArray as FsciArrayApiCoreArray, CoreArrayBackend as FsciArrayApiCoreArrayBackend,
    CreationRequest as ArrayApiCreationRequest, DType as FsciArrayApiDType,
    ExecutionMode as FsciArrayApiExecutionMode, FullRequest as ArrayApiFullRequest,
    IndexExpr as FsciArrayApiIndexExpr, IndexRequest as ArrayApiIndexRequest,
    IndexingMode as FsciArrayApiIndexingMode, LinspaceRequest as ArrayApiLinspaceRequest,
    MemoryOrder as FsciArrayApiMemoryOrder, ScalarValue as FsciArrayApiScalarValue,
    Shape as FsciArrayApiShape, SliceSpec as FsciArrayApiSliceSpec, arange as arrayapi_arange,
    arange_with_audit as arrayapi_arange_with_audit, broadcast_shapes as arrayapi_broadcast_shapes,
    broadcast_shapes_with_audit as arrayapi_broadcast_shapes_with_audit,
    from_slice as arrayapi_from_slice, from_slice_with_audit as arrayapi_from_slice_with_audit,
    full as arrayapi_full, getitem as arrayapi_getitem,
    getitem_with_audit as arrayapi_getitem_with_audit, linspace as arrayapi_linspace,
    ones as arrayapi_ones, reshape as arrayapi_reshape,
    reshape_with_audit as arrayapi_reshape_with_audit, result_type as arrayapi_result_type,
    sync_audit_ledger as array_api_sync_audit_ledger, transpose as arrayapi_transpose,
    zeros as arrayapi_zeros,
};
use fsci_fft::sync_audit_ledger as fft_sync_audit_ledger;
use fsci_integrate::{
    CubatureOptions, DblquadOptions, QuadOptions, ToleranceValue, cubature, cubature_scalar,
    cumulative_simpson, cumulative_trapezoid, dblquad, fixed_quad, gauss_legendre, newton_cotes,
    qmc_quad, quad, quad_vec, romb, simpson, sync_audit_ledger as integrate_sync_audit_ledger,
    tplquad, trapezoid, validate_tol, validate_tol_with_audit,
};
use fsci_interpolate::{
    BSpline, CubicSplineStandalone, Interp1d, Interp1dOptions, InterpKind as FsciInterpKind,
    RegularGridInterpolator, RegularGridMethod as FsciRegularGridMethod, SplineBc as FsciSplineBc,
};
use fsci_io::{MatArray, loadmat, loadtxt, mmread, mmwrite, savemat, savetxt, wav_read, wav_write};
use fsci_linalg::{
    DecompOptions, InvOptions, LinalgError, LstsqDriver, LstsqOptions, MatrixAssumption,
    PinvOptions, SolveOptions, TriangularSolveOptions, TriangularTranspose, cholesky, det,
    det_with_audit, eig, eigh, inv, inv_with_audit, lstsq, lstsq_with_audit, pinv, pinv_with_audit,
    qr, solve, solve_banded, solve_banded_with_audit, solve_triangular,
    solve_triangular_with_audit, solve_with_audit, svd,
    sync_audit_ledger as linalg_sync_audit_ledger,
};
use fsci_ndimage::{
    BoundaryMode as FsciNdimageBoundaryMode, NdArray as FsciNdimageArray,
    binary_dilation as ndimage_binary_dilation, binary_erosion as ndimage_binary_erosion,
    distance_transform_edt as ndimage_distance_transform_edt,
    gaussian_filter as ndimage_gaussian_filter, label as ndimage_label,
};
use fsci_opt::sync_audit_ledger as opt_sync_audit_ledger;
use fsci_opt::{
    BasinhoppingOptions, ConvergenceStatus as OptConvergenceStatus, DifferentialEvolutionOptions,
    MinimizeOptions, OptError, OptimizeMethod, RootMethod, RootOptions, basinhopping, brute,
    differential_evolution, dual_annealing, minimize, minimize_with_audit, root_scalar,
};
use fsci_runtime::{AuditLedger, RuntimeMode, SolverPortfolio};
use fsci_special::{
    Complex64, SpecialError as FsciSpecialError, SpecialErrorKind as FsciSpecialErrorKind,
    SpecialTensor as FsciSpecialTensor, bdtr as special_bdtr, bdtrc as special_bdtrc,
    bdtri as special_bdtri, bei as special_bei, ber as special_ber, beta as special_beta,
    betainc as special_betainc, betaln as special_betaln,
    boxcox_transform_scalar as special_boxcox_transform, boxcox1p_scalar as special_boxcox1p,
    btdtr as special_btdtr, btdtrc as special_btdtrc, btdtri as special_btdtri,
    btdtria as special_btdtria, btdtrib as special_btdtrib, cbrt as special_cbrt,
    chdtr as special_chdtr, chdtrc as special_chdtrc, chdtri as special_chdtri,
    chdtriv as special_chdtriv, comb as special_comb, cosdg as special_cosdg,
    cotdg as special_cotdg, dawsn as special_dawsn, digamma as special_digamma,
    ellipe as special_ellipe, ellipeinc as special_ellipeinc, ellipj as special_ellipj,
    ellipk as special_ellipk, ellipkinc as special_ellipkinc, ellipkm1 as special_ellipkm1,
    entr as special_entr, erf as special_erf, erfc as special_erfc, erfcinv as special_erfcinv,
    erfcx as special_erfcx, erfi as special_erfi, erfinv as special_erfinv,
    eval_chebyt as special_eval_chebyt, eval_chebyu as special_eval_chebyu,
    eval_gegenbauer as special_eval_gegenbauer, eval_genlaguerre as special_eval_genlaguerre,
    eval_hermite as special_eval_hermite, eval_hermitenorm as special_eval_hermitenorm,
    eval_jacobi as special_eval_jacobi, eval_laguerre as special_eval_laguerre,
    eval_legendre as special_eval_legendre, eval_sh_chebyt as special_eval_sh_chebyt,
    eval_sh_chebyu as special_eval_sh_chebyu, eval_sh_legendre as special_eval_sh_legendre,
    exp1 as special_exp1, exp2 as special_exp2, exp10 as special_exp10, expi as special_expi,
    expit as special_expit, expm1 as special_expm1, expn_scalar as special_expn,
    exprel as special_exprel, factorial as special_factorial, factorial2 as special_factorial2,
    fdtr as special_fdtr, fdtrc as special_fdtrc, fdtri as special_fdtri,
    fdtridfd as special_fdtridfd, fresnel as special_fresnel, gamma as special_gamma,
    gammainc as special_gammainc, gammaincc as special_gammaincc, gammaln as special_gammaln,
    gdtr as special_gdtr, gdtrc as special_gdtrc, gdtria as special_gdtria,
    gdtrib as special_gdtrib, gdtrix as special_gdtrix, huber as special_huber,
    hurwitz_zeta as special_hurwitz_zeta, hyp0f1 as special_hyp0f1, hyp1f1 as special_hyp1f1,
    hyp2f1 as special_hyp2f1, inv_boxcox_scalar as special_inv_boxcox,
    inv_boxcox1p_scalar as special_inv_boxcox1p, iv as special_iv, ivp as special_ivp,
    j0 as special_j0, j1 as special_j1, jn as special_jn, jvp as special_jvp, kei as special_kei,
    ker as special_ker, kl_div as special_kl_div, kolmogi as special_kolmogi,
    kolmogorov as special_kolmogorov, kv as special_kv, kvp as special_kvp,
    lambertw as special_lambertw, log_ndtr_scalar as special_log_ndtr_scalar,
    log1p as special_log1p, logaddexp as special_logaddexp, logaddexp2 as special_logaddexp2,
    logit as special_logit, logsumexp as special_logsumexp, lpmv as special_lpmv,
    modstruve as special_modstruve, multigammaln as special_multigammaln, nbdtr as special_nbdtr,
    nbdtrc as special_nbdtrc, nbdtri as special_nbdtri, ndtr as special_ndtr,
    ndtri as special_ndtri, nrdtrimn as special_nrdtrimn, nrdtrisd as special_nrdtrisd,
    owens_t as special_owens_t, pdtr as special_pdtr, pdtrc as special_pdtrc,
    pdtri as special_pdtri, pdtrik as special_pdtrik, perm as special_perm, poch as special_poch,
    polygamma as special_polygamma, pseudo_huber as special_pseudo_huber, radian as special_radian,
    rel_entr as special_rel_entr, rgamma as special_rgamma, roots_chebyt as special_roots_chebyt,
    roots_chebyu as special_roots_chebyu, roots_gegenbauer as special_roots_gegenbauer,
    roots_genlaguerre as special_roots_genlaguerre, roots_hermite as special_roots_hermite,
    roots_hermitenorm as special_roots_hermitenorm, roots_jacobi as special_roots_jacobi,
    roots_laguerre as special_roots_laguerre, roots_legendre as special_roots_legendre,
    shichi as special_shichi, sici as special_sici, sinc as special_sinc, sindg as special_sindg,
    softplus as special_softplus, spence as special_spence, sph_harm_y as special_sph_harm_y,
    spherical_in as special_spherical_in, spherical_jn as special_spherical_jn,
    spherical_kn as special_spherical_kn, spherical_yn as special_spherical_yn,
    stdtr as special_stdtr, stdtrc as special_stdtrc, stdtridf as special_stdtridf,
    stdtrit as special_stdtrit, struve as special_struve, tandg as special_tandg,
    wright_bessel as special_wright_bessel, xlog1py as special_xlog1py, xlogx as special_xlogx,
    xlogy as special_xlogy, y0 as special_y0, y1 as special_y1, yn as special_yn,
    yvp as special_yvp, zeta as special_zeta, zetac as special_zetac,
};
#[cfg(feature = "dashboard")]
use ftui::{PackedRgba, Style};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::any::Any;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct HarnessConfig {
    pub oracle_root: PathBuf,
    pub fixture_root: PathBuf,
    pub strict_mode: bool,
}

impl HarnessConfig {
    #[must_use]
    pub fn default_paths() -> Self {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        Self {
            oracle_root: repo_root.join("legacy_scipy_code/scipy"),
            fixture_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"),
            strict_mode: true,
        }
    }

    #[must_use]
    pub fn artifact_dir_for(&self, packet_id: &str) -> PathBuf {
        self.fixture_root.join("artifacts").join(packet_id)
    }
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self::default_paths()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessReport {
    pub suite: &'static str,
    pub oracle_present: bool,
    pub cases_run: usize,
    pub failed_cases: usize,
    pub strict_mode: bool,
}

#[derive(Debug, Error)]
pub enum HarnessError {
    #[error("fixture load failed for {path}: {source}")]
    FixtureIo { path: PathBuf, source: io::Error },
    #[error("fixture parse failed for {path}: {source}")]
    FixtureParse {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("artifact write failed for {path}: {source}")]
    ArtifactIo { path: PathBuf, source: io::Error },
    #[error("raptorq sidecar generation failed: {0}")]
    RaptorQ(String),
    #[error("serialization failed: {0}")]
    Serialization(String),
    #[error("audit ledger mutex poisoned: {0}")]
    AuditLedgerPoisoned(String),
    #[error("linalg execution failed: {0}")]
    Linalg(#[from] LinalgError),
    #[error("optimize execution failed: {0}")]
    Optimize(#[from] OptError),
    // Typed forwarding for the remaining fsci-* crate errors per
    // frankenscipy-rkzk. Consumers can `matches!(err, HarnessError::Special(_))`
    // instead of string-parsing a RaptorQ/Serialization blob.
    #[error("special function error: {0}")]
    Special(#[from] fsci_special::SpecialError),
    #[error("fft execution failed: {0}")]
    Fft(#[from] fsci_fft::FftError),
    #[error("sparse execution failed: {0}")]
    Sparse(#[from] fsci_sparse::SparseError),
    #[error("integrate validation failed: {0}")]
    Integrate(#[from] fsci_integrate::IntegrateValidationError),
    #[error("cluster execution failed: {0}")]
    Cluster(#[from] fsci_cluster::ClusterError),
    #[error("spatial execution failed: {0}")]
    Spatial(#[from] fsci_spatial::SpatialError),
    #[error("signal execution failed: {0}")]
    Signal(#[from] fsci_signal::SignalError),
    #[error("array-api execution failed: {0}")]
    ArrayApi(#[from] fsci_arrayapi::ArrayApiError),
    #[error("interpolate execution failed: {0}")]
    Interpolate(#[from] fsci_interpolate::InterpError),
    #[error("ndimage execution failed: {0}")]
    Ndimage(#[from] fsci_ndimage::NdimageError),
    #[error("io execution failed: {0}")]
    Io(#[from] fsci_io::IoError),
    #[error("failed to launch python oracle `{python_bin}`: {source}")]
    PythonLaunch {
        python_bin: String,
        source: io::Error,
    },
    #[error("python oracle script not found at {path}")]
    PythonScriptMissing { path: PathBuf },
    #[error("python oracle `{python_bin}` failed: {stderr}")]
    PythonFailed { python_bin: String, stderr: String },
    #[error("python oracle requires scipy but it is unavailable: {stderr}")]
    PythonSciPyMissing { stderr: String },
    /// A Python module required by the oracle (other than scipy) was
    /// missing at import time — typically numpy or sklearn. Broken out
    /// from PythonFailed (br-wbzg) so triage can distinguish install
    /// problems from script bugs.
    #[error("python oracle is missing module `{module}`: {stderr}")]
    PythonModuleMissing { module: String, stderr: String },
    #[error("oracle capture parse failed for {path}: {source}")]
    OracleParse {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("oracle capture mismatch: {detail}")]
    OracleCaptureMismatch { detail: String },
}

/// Classify a Python interpreter stderr blob into a `HarnessError`.
/// Centralizes the `No module named ...` parsing so every oracle call
/// site distinguishes install-time from script-time failures (br-wbzg).
///
/// Matches Python's two common "module missing" phrasings:
///
/// - `ModuleNotFoundError: No module named 'scipy'`
/// - `ImportError: No module named scipy`
///
/// and special-cases scipy to return `PythonSciPyMissing` for
/// backward compatibility with existing callers. numpy / sklearn /
/// any other missing module routes to `PythonModuleMissing`.
fn classify_python_stderr(python_bin: String, stderr: String) -> HarnessError {
    if let Some(module) = extract_missing_module(&stderr) {
        if module == "scipy" {
            return HarnessError::PythonSciPyMissing { stderr };
        }
        return HarnessError::PythonModuleMissing { module, stderr };
    }
    HarnessError::PythonFailed { python_bin, stderr }
}

/// Extract the module name from a `No module named <x>` fragment in
/// `stderr`. Returns `None` when no such fragment is present.
fn extract_missing_module(stderr: &str) -> Option<String> {
    // Look for "No module named" and take the next word (strip quotes,
    // take up to the first non-identifier char for dotted imports we
    // split on the first `.`).
    let needle = "No module named ";
    let idx = stderr.find(needle)?;
    let rest = &stderr[idx + needle.len()..];
    let rest = rest.trim_start_matches(['\'', '"']);
    let end = rest
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '.'))
        .unwrap_or(rest.len());
    let raw = &rest[..end];
    if raw.is_empty() {
        return None;
    }
    Some(raw.split('.').next().unwrap_or(raw).to_owned())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum FixtureToleranceValue {
    Scalar(f64),
    Vector(Vec<f64>),
}

impl From<FixtureToleranceValue> for ToleranceValue {
    fn from(value: FixtureToleranceValue) -> Self {
        match value {
            FixtureToleranceValue::Scalar(v) => Self::Scalar(v),
            FixtureToleranceValue::Vector(vs) => Self::Vector(vs),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExpectedOutcome {
    Ok {
        rtol: FixtureToleranceValue,
        atol: FixtureToleranceValue,
        warning_rtol_clamped: bool,
    },
    Error {
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ValidateTolCase {
    pub case_id: String,
    pub mode: RuntimeMode,
    pub n: usize,
    pub rtol: FixtureToleranceValue,
    pub atol: FixtureToleranceValue,
    pub expected: ExpectedOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<ValidateTolCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FixtureMatrixAssumption {
    General,
    Diagonal,
    UpperTriangular,
    LowerTriangular,
    Symmetric,
    Hermitian,
    PositiveDefinite,
}

impl From<FixtureMatrixAssumption> for MatrixAssumption {
    fn from(value: FixtureMatrixAssumption) -> Self {
        match value {
            FixtureMatrixAssumption::General => Self::General,
            FixtureMatrixAssumption::Diagonal => Self::Diagonal,
            FixtureMatrixAssumption::UpperTriangular => Self::UpperTriangular,
            FixtureMatrixAssumption::LowerTriangular => Self::LowerTriangular,
            FixtureMatrixAssumption::Symmetric => Self::Symmetric,
            FixtureMatrixAssumption::Hermitian => Self::Hermitian,
            FixtureMatrixAssumption::PositiveDefinite => Self::PositiveDefinite,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FixtureTriangularTranspose {
    NoTranspose,
    Transpose,
    ConjugateTranspose,
}

impl From<FixtureTriangularTranspose> for TriangularTranspose {
    fn from(value: FixtureTriangularTranspose) -> Self {
        match value {
            FixtureTriangularTranspose::NoTranspose => Self::NoTranspose,
            FixtureTriangularTranspose::Transpose => Self::Transpose,
            FixtureTriangularTranspose::ConjugateTranspose => Self::ConjugateTranspose,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FixtureQrMode {
    Full,
    Economic,
    R,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LinalgExpectedOutcome {
    Vector {
        #[serde(
            deserialize_with = "deserialize_maybe_nan_required_vec",
            serialize_with = "serialize_maybe_nan_required_vec"
        )]
        values: Vec<f64>,
        atol: f64,
        rtol: f64,
        #[serde(default)]
        expect_warning_ill_conditioned: Option<bool>,
    },
    Matrix {
        #[serde(
            deserialize_with = "deserialize_maybe_nan_matrix",
            serialize_with = "serialize_maybe_nan_matrix"
        )]
        values: Vec<Vec<f64>>,
        atol: f64,
        rtol: f64,
    },
    Scalar {
        #[serde(with = "maybe_nan_f64")]
        value: f64,
        atol: f64,
        rtol: f64,
    },
    Lstsq {
        #[serde(
            deserialize_with = "deserialize_maybe_nan_required_vec",
            serialize_with = "serialize_maybe_nan_required_vec"
        )]
        x: Vec<f64>,
        #[serde(
            deserialize_with = "deserialize_maybe_nan_required_vec",
            serialize_with = "serialize_maybe_nan_required_vec"
        )]
        residuals: Vec<f64>,
        rank: usize,
        #[serde(
            deserialize_with = "deserialize_maybe_nan_required_vec",
            serialize_with = "serialize_maybe_nan_required_vec"
        )]
        singular_values: Vec<f64>,
        atol: f64,
        rtol: f64,
    },
    Pinv {
        #[serde(
            deserialize_with = "deserialize_maybe_nan_matrix",
            serialize_with = "serialize_maybe_nan_matrix"
        )]
        values: Vec<Vec<f64>>,
        rank: usize,
        atol: f64,
        rtol: f64,
    },
    Error {
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum LinalgCase {
    Solve {
        case_id: String,
        mode: RuntimeMode,
        #[serde(
            deserialize_with = "deserialize_maybe_nan_matrix",
            serialize_with = "serialize_maybe_nan_matrix"
        )]
        a: Vec<Vec<f64>>,
        #[serde(
            deserialize_with = "deserialize_maybe_nan_required_vec",
            serialize_with = "serialize_maybe_nan_required_vec"
        )]
        b: Vec<f64>,
        assume_a: Option<FixtureMatrixAssumption>,
        #[serde(default)]
        lower: Option<bool>,
        #[serde(default)]
        transposed: Option<bool>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
    SolveTriangular {
        case_id: String,
        mode: RuntimeMode,
        #[serde(
            deserialize_with = "deserialize_maybe_nan_matrix",
            serialize_with = "serialize_maybe_nan_matrix"
        )]
        a: Vec<Vec<f64>>,
        #[serde(
            deserialize_with = "deserialize_maybe_nan_required_vec",
            serialize_with = "serialize_maybe_nan_required_vec"
        )]
        b: Vec<f64>,
        trans: Option<FixtureTriangularTranspose>,
        #[serde(default)]
        lower: Option<bool>,
        #[serde(default)]
        unit_diagonal: Option<bool>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
    SolveBanded {
        case_id: String,
        mode: RuntimeMode,
        l_and_u: [usize; 2],
        #[serde(
            deserialize_with = "deserialize_maybe_nan_matrix",
            serialize_with = "serialize_maybe_nan_matrix"
        )]
        ab: Vec<Vec<f64>>,
        #[serde(
            deserialize_with = "deserialize_maybe_nan_required_vec",
            serialize_with = "serialize_maybe_nan_required_vec"
        )]
        b: Vec<f64>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
    Inv {
        case_id: String,
        mode: RuntimeMode,
        #[serde(
            deserialize_with = "deserialize_maybe_nan_matrix",
            serialize_with = "serialize_maybe_nan_matrix"
        )]
        a: Vec<Vec<f64>>,
        assume_a: Option<FixtureMatrixAssumption>,
        #[serde(default)]
        lower: Option<bool>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
    Det {
        case_id: String,
        mode: RuntimeMode,
        #[serde(
            deserialize_with = "deserialize_maybe_nan_matrix",
            serialize_with = "serialize_maybe_nan_matrix"
        )]
        a: Vec<Vec<f64>>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
    Lstsq {
        case_id: String,
        mode: RuntimeMode,
        #[serde(
            deserialize_with = "deserialize_maybe_nan_matrix",
            serialize_with = "serialize_maybe_nan_matrix"
        )]
        a: Vec<Vec<f64>>,
        #[serde(
            deserialize_with = "deserialize_maybe_nan_required_vec",
            serialize_with = "serialize_maybe_nan_required_vec"
        )]
        b: Vec<f64>,
        #[serde(default)]
        #[serde(
            deserialize_with = "deserialize_maybe_nan_option_f64",
            serialize_with = "serialize_maybe_nan_option_f64"
        )]
        cond: Option<f64>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
    Pinv {
        case_id: String,
        mode: RuntimeMode,
        #[serde(
            deserialize_with = "deserialize_maybe_nan_matrix",
            serialize_with = "serialize_maybe_nan_matrix"
        )]
        a: Vec<Vec<f64>>,
        #[serde(default)]
        #[serde(
            deserialize_with = "deserialize_maybe_nan_option_f64",
            serialize_with = "serialize_maybe_nan_option_f64"
        )]
        atol: Option<f64>,
        #[serde(default)]
        #[serde(
            deserialize_with = "deserialize_maybe_nan_option_f64",
            serialize_with = "serialize_maybe_nan_option_f64"
        )]
        rtol: Option<f64>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
    Qr {
        case_id: String,
        mode: RuntimeMode,
        #[serde(
            deserialize_with = "deserialize_maybe_nan_matrix",
            serialize_with = "serialize_maybe_nan_matrix"
        )]
        a: Vec<Vec<f64>>,
        #[serde(default)]
        qr_mode: Option<FixtureQrMode>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
    Svd {
        case_id: String,
        mode: RuntimeMode,
        #[serde(
            deserialize_with = "deserialize_maybe_nan_matrix",
            serialize_with = "serialize_maybe_nan_matrix"
        )]
        a: Vec<Vec<f64>>,
        #[serde(default)]
        compute_uv: Option<bool>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
    Cholesky {
        case_id: String,
        mode: RuntimeMode,
        #[serde(
            deserialize_with = "deserialize_maybe_nan_matrix",
            serialize_with = "serialize_maybe_nan_matrix"
        )]
        a: Vec<Vec<f64>>,
        #[serde(default)]
        lower: Option<bool>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
    Eig {
        case_id: String,
        mode: RuntimeMode,
        #[serde(
            deserialize_with = "deserialize_maybe_nan_matrix",
            serialize_with = "serialize_maybe_nan_matrix"
        )]
        a: Vec<Vec<f64>>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
    Eigh {
        case_id: String,
        mode: RuntimeMode,
        #[serde(
            deserialize_with = "deserialize_maybe_nan_matrix",
            serialize_with = "serialize_maybe_nan_matrix"
        )]
        a: Vec<Vec<f64>>,
        #[serde(default)]
        lower: Option<bool>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
}

impl LinalgCase {
    fn case_id(&self) -> &str {
        match self {
            Self::Solve { case_id, .. }
            | Self::SolveTriangular { case_id, .. }
            | Self::SolveBanded { case_id, .. }
            | Self::Inv { case_id, .. }
            | Self::Det { case_id, .. }
            | Self::Lstsq { case_id, .. }
            | Self::Pinv { case_id, .. }
            | Self::Qr { case_id, .. }
            | Self::Svd { case_id, .. }
            | Self::Cholesky { case_id, .. }
            | Self::Eig { case_id, .. }
            | Self::Eigh { case_id, .. } => case_id,
        }
    }

    fn expected(&self) -> &LinalgExpectedOutcome {
        match self {
            Self::Solve { expected, .. }
            | Self::SolveTriangular { expected, .. }
            | Self::SolveBanded { expected, .. }
            | Self::Inv { expected, .. }
            | Self::Det { expected, .. }
            | Self::Lstsq { expected, .. }
            | Self::Pinv { expected, .. }
            | Self::Qr { expected, .. }
            | Self::Svd { expected, .. }
            | Self::Cholesky { expected, .. }
            | Self::Eig { expected, .. }
            | Self::Eigh { expected, .. } => expected,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LinalgPacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<LinalgCase>,
    /// Optional provenance metadata captured alongside the cases by
    /// the fixture-generation tooling. Shape is intentionally opaque
    /// (serde_json::Value) — we only carry it end-to-end so consumers
    /// that want to inspect it can, without breaking the strict
    /// `deny_unknown_fields` contract on the packet. See
    /// `fixtures/PROVENANCE.md` for the human-readable counterpart.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oracle_provenance: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizeMinObjective {
    Rosenbrock2,
    Ackley2,
    Rastrigin2,
    Himmelblau2,
    Sin1d,
    Sphere2,
    ShiftedQuadratic,
    TranslatedQuadratic,
    ScaledQuadratic,
    RotatedQuadratic,
    L1Nonsmooth,
    FlatQuartic,
    NanBranch,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizeRootObjective {
    CubicMinusTwo,
    CosMinusX,
    SinMinusHalf,
    LinearShift03,
    NanBranch,
    StepDiscontinuous,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OptimizeExpectedOutcome {
    MinimizePoint {
        x: Vec<f64>,
        fun: f64,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
        #[serde(default)]
        require_success: Option<bool>,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    RootValue {
        root: f64,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
        #[serde(default)]
        require_converged: Option<bool>,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    MinimizeStatus {
        status: OptConvergenceStatus,
        success: bool,
    },
    RootStatus {
        status: OptConvergenceStatus,
        converged: bool,
    },
    Error {
        error: String,
    },
}

// br-7470: constraint specification for differential_evolution. Tagged
// by `kind`. `linear`: A*x ∈ [lb, ub], where A is a 2D matrix. `nonlinear`:
// f(x) ∈ [lb, ub], where f is a named integrand from the registry. Both
// flavors are translated into a single violation closure that returns
// the sum of squared boundary excess across all constraints.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DeConstraintSpec {
    Linear {
        #[serde(rename = "A")]
        a: Vec<Vec<f64>>,
        lb: Vec<f64>,
        ub: Vec<f64>,
    },
    Nonlinear {
        fn_name: String,
        lb: Vec<f64>,
        ub: Vec<f64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum OptimizeCase {
    Minimize {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        method: OptimizeMethod,
        objective: OptimizeMinObjective,
        x0: Vec<f64>,
        #[serde(default)]
        seed: Option<u64>,
        #[serde(default)]
        hessp: bool,
        #[serde(default)]
        tol: Option<f64>,
        #[serde(default)]
        maxiter: Option<usize>,
        #[serde(default)]
        maxfev: Option<usize>,
        expected: OptimizeExpectedOutcome,
    },
    Root {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        method: RootMethod,
        objective: OptimizeRootObjective,
        bracket: [f64; 2],
        #[serde(default)]
        xtol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
        #[serde(default)]
        maxiter: Option<usize>,
        expected: OptimizeExpectedOutcome,
    },
    DifferentialEvolution {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        objective: OptimizeMinObjective,
        bounds: Vec<[f64; 2]>,
        #[serde(default)]
        seed: Option<u64>,
        #[serde(default)]
        tol: Option<f64>,
        #[serde(default)]
        maxiter: Option<usize>,
        #[serde(default)]
        popsize: Option<usize>,
        // br-7470: optional constraint set. Empty -> unconstrained DE
        // (preserves existing case behavior). Non-empty -> penalty-based
        // constrained DE via fsci_opt::differential_evolution_constrained.
        #[serde(default)]
        constraints: Vec<DeConstraintSpec>,
        expected: OptimizeExpectedOutcome,
    },
    Basinhopping {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        objective: OptimizeMinObjective,
        x0: Vec<f64>,
        #[serde(default)]
        seed: Option<u64>,
        #[serde(default)]
        maxiter: Option<usize>,
        #[serde(default)]
        tol: Option<f64>,
        expected: OptimizeExpectedOutcome,
    },
    DualAnnealing {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        objective: OptimizeMinObjective,
        bounds: Vec<[f64; 2]>,
        #[serde(default)]
        seed: Option<u64>,
        #[serde(default)]
        maxiter: Option<usize>,
        expected: OptimizeExpectedOutcome,
    },
    Brute {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        objective: OptimizeMinObjective,
        ranges: Vec<[f64; 2]>,
        #[serde(default)]
        ns: Option<usize>,
        #[serde(default)]
        seed: Option<u64>,
        expected: OptimizeExpectedOutcome,
    },
}

impl OptimizeCase {
    fn case_id(&self) -> &str {
        match self {
            Self::Minimize { case_id, .. }
            | Self::Root { case_id, .. }
            | Self::DifferentialEvolution { case_id, .. }
            | Self::Basinhopping { case_id, .. }
            | Self::DualAnnealing { case_id, .. }
            | Self::Brute { case_id, .. } => case_id,
        }
    }

    #[cfg(test)]
    fn category(&self) -> &str {
        match self {
            Self::Minimize { category, .. }
            | Self::Root { category, .. }
            | Self::DifferentialEvolution { category, .. }
            | Self::Basinhopping { category, .. }
            | Self::DualAnnealing { category, .. }
            | Self::Brute { category, .. } => category,
        }
    }

    fn expected(&self) -> &OptimizeExpectedOutcome {
        match self {
            Self::Minimize { expected, .. }
            | Self::Root { expected, .. }
            | Self::DifferentialEvolution { expected, .. }
            | Self::Basinhopping { expected, .. }
            | Self::DualAnnealing { expected, .. }
            | Self::Brute { expected, .. } => expected,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OptimizePacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<OptimizeCase>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InterpolateInterpKind {
    Linear,
    Nearest,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InterpolateRegularGridMethod {
    Linear,
    Nearest,
    Pchip,
    Cubic,
    Quintic,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InterpolateSplineBc {
    Natural,
    NotAKnot,
    Clamped {
        #[serde(with = "maybe_nan_f64")]
        left_derivative: f64,
        #[serde(with = "maybe_nan_f64")]
        right_derivative: f64,
    },
    Periodic,
    Tuple {
        left_order: usize,
        #[serde(with = "maybe_nan_f64")]
        left_value: f64,
        right_order: usize,
        #[serde(with = "maybe_nan_f64")]
        right_value: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InterpolateExpectedOutcome {
    Vector {
        values: Vec<f64>,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
    },
    Error {
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum InterpolateCase {
    Interp1d {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        kind: InterpolateInterpKind,
        x: Vec<f64>,
        y: Vec<f64>,
        x_new: Vec<f64>,
        #[serde(default)]
        bounds_error: Option<bool>,
        #[serde(default)]
        fill_value: Option<f64>,
        expected: InterpolateExpectedOutcome,
    },
    RegularGridInterpolator {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        method: InterpolateRegularGridMethod,
        points: Vec<Vec<f64>>,
        values: Vec<f64>,
        xi: Vec<Vec<f64>>,
        #[serde(default)]
        bounds_error: Option<bool>,
        #[serde(default)]
        fill_value: Option<f64>,
        expected: InterpolateExpectedOutcome,
    },
    CubicSpline {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        x: Vec<f64>,
        y: Vec<f64>,
        x_new: Vec<f64>,
        #[serde(default)]
        bc: Option<InterpolateSplineBc>,
        expected: InterpolateExpectedOutcome,
    },
    #[serde(rename = "bspline")]
    BSpline {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        knots: Vec<f64>,
        coefficients: Vec<f64>,
        degree: usize,
        x_new: Vec<f64>,
        expected: InterpolateExpectedOutcome,
    },
}

impl InterpolateCase {
    fn case_id(&self) -> &str {
        match self {
            Self::Interp1d { case_id, .. }
            | Self::RegularGridInterpolator { case_id, .. }
            | Self::CubicSpline { case_id, .. }
            | Self::BSpline { case_id, .. } => case_id,
        }
    }

    fn expected(&self) -> &InterpolateExpectedOutcome {
        match self {
            Self::Interp1d { expected, .. }
            | Self::RegularGridInterpolator { expected, .. }
            | Self::CubicSpline { expected, .. }
            | Self::BSpline { expected, .. } => expected,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InterpolatePacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<InterpolateCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IoExpectedOutcome {
    Matrix {
        rows: usize,
        cols: usize,
        values: Vec<f64>,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
    },
    Wav {
        sample_rate: u32,
        channels: u16,
        bits_per_sample: u16,
        values: Vec<f64>,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
    },
    Error {
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum IoCase {
    Mmread {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        content: String,
        expected: IoExpectedOutcome,
    },
    Mmwrite {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        rows: usize,
        cols: usize,
        data: Vec<f64>,
        expected: IoExpectedOutcome,
    },
    Loadmat {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        content_hex: String,
        expected: IoExpectedOutcome,
    },
    Savemat {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        name: String,
        rows: usize,
        cols: usize,
        data: Vec<f64>,
        expected: IoExpectedOutcome,
    },
    Loadtxt {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        content: String,
        expected: IoExpectedOutcome,
    },
    Savetxt {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        rows: usize,
        cols: usize,
        data: Vec<f64>,
        delimiter: String,
        expected: IoExpectedOutcome,
    },
    WavWrite {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        sample_rate: u32,
        channels: u16,
        data: Vec<f64>,
        expected: IoExpectedOutcome,
    },
}

impl IoCase {
    fn case_id(&self) -> &str {
        match self {
            Self::Mmread { case_id, .. }
            | Self::Mmwrite { case_id, .. }
            | Self::Loadmat { case_id, .. }
            | Self::Savemat { case_id, .. }
            | Self::Loadtxt { case_id, .. }
            | Self::Savetxt { case_id, .. }
            | Self::WavWrite { case_id, .. } => case_id,
        }
    }

    fn expected(&self) -> &IoExpectedOutcome {
        match self {
            Self::Mmread { expected, .. }
            | Self::Mmwrite { expected, .. }
            | Self::Loadmat { expected, .. }
            | Self::Savemat { expected, .. }
            | Self::Loadtxt { expected, .. }
            | Self::Savetxt { expected, .. }
            | Self::WavWrite { expected, .. } => expected,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct IoPacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<IoCase>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NdimageBoundaryMode {
    Reflect,
    Constant,
    Nearest,
    Wrap,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NdimageExpectedOutcome {
    Array {
        values: Vec<f64>,
        shape: Vec<usize>,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
    },
    Label {
        labels: Vec<f64>,
        shape: Vec<usize>,
        num_features: usize,
    },
    Error {
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum NdimageCase {
    GaussianFilter {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        input: Vec<f64>,
        shape: Vec<usize>,
        sigma: f64,
        boundary: NdimageBoundaryMode,
        #[serde(default)]
        cval: f64,
        expected: NdimageExpectedOutcome,
    },
    Label {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        input: Vec<f64>,
        shape: Vec<usize>,
        expected: NdimageExpectedOutcome,
    },
    BinaryErosion {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        input: Vec<f64>,
        shape: Vec<usize>,
        structure_size: usize,
        iterations: usize,
        expected: NdimageExpectedOutcome,
    },
    BinaryDilation {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        input: Vec<f64>,
        shape: Vec<usize>,
        structure_size: usize,
        iterations: usize,
        expected: NdimageExpectedOutcome,
    },
    DistanceTransformEdt {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        input: Vec<f64>,
        shape: Vec<usize>,
        #[serde(default)]
        sampling: Option<Vec<f64>>,
        expected: NdimageExpectedOutcome,
    },
}

impl NdimageCase {
    fn case_id(&self) -> &str {
        match self {
            Self::GaussianFilter { case_id, .. }
            | Self::Label { case_id, .. }
            | Self::BinaryErosion { case_id, .. }
            | Self::BinaryDilation { case_id, .. }
            | Self::DistanceTransformEdt { case_id, .. } => case_id,
        }
    }

    fn expected(&self) -> &NdimageExpectedOutcome {
        match self {
            Self::GaussianFilter { expected, .. }
            | Self::Label { expected, .. }
            | Self::BinaryErosion { expected, .. }
            | Self::BinaryDilation { expected, .. }
            | Self::DistanceTransformEdt { expected, .. } => expected,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NdimagePacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<NdimageCase>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SpecialCaseFunction {
    Gamma,
    Gammaln,
    Multigammaln,
    Digamma,
    Polygamma,
    Rgamma,
    Gammainc,
    Gammaincc,
    Factorial,
    Factorial2,
    Comb,
    Perm,
    Poch,
    Erf,
    Erfc,
    Erfinv,
    Erfcinv,
    Erfcx,
    Erfi,
    OwensT,
    Beta,
    Betaln,
    Betainc,
    Btdtr,
    Btdtrc,
    Btdtri,
    Btdtria,
    Btdtrib,
    Fdtr,
    Fdtrc,
    Fdtri,
    Fdtridfd,
    Stdtr,
    Stdtrc,
    Stdtridf,
    Stdtrit,
    Bdtr,
    Bdtrc,
    Bdtri,
    Nbdtr,
    Nbdtrc,
    Nbdtri,
    Pdtr,
    Pdtrc,
    Pdtri,
    Pdtrik,
    Chdtr,
    Chdtrc,
    Chdtri,
    Chdtriv,
    Gdtr,
    Gdtrc,
    Gdtria,
    Gdtrib,
    Gdtrix,
    Ellipk,
    Ellipkm1,
    Ellipe,
    Ellipkinc,
    Ellipeinc,
    EllipjSn,
    EllipjCn,
    EllipjDn,
    EllipjPh,
    Lambertw,
    Exp1,
    Expi,
    Expn,
    Hyp0f1,
    Hyp1f1,
    Hyp2f1,
    J0,
    J1,
    Jn,
    Y0,
    Y1,
    Yn,
    Iv,
    Kv,
    WrightBessel,
    Jvp,
    Yvp,
    Ivp,
    Kvp,
    SphericalJn,
    SphericalYn,
    SphericalIn,
    SphericalKn,
    Sinc,
    Xlogy,
    Xlog1py,
    Xlogx,
    Kolmogorov,
    Kolmogi,
    Entr,
    RelEntr,
    KlDiv,
    Ndtr,
    Ndtri,
    Nrdtrimn,
    Nrdtrisd,
    LogNdtr,
    Logsumexp,
    Log1p,
    Expm1,
    Logaddexp,
    Logaddexp2,
    Softplus,
    Huber,
    PseudoHuber,
    Cosdg,
    Sindg,
    Tandg,
    Cotdg,
    Radian,
    Cbrt,
    Exp2,
    Exp10,
    Boxcox,
    InvBoxcox,
    Boxcox1p,
    InvBoxcox1p,
    FresnelS,
    FresnelC,
    Dawsn,
    SiciSi,
    SiciCi,
    ShichiShi,
    ShichiChi,
    Struve,
    Modstruve,
    Ber,
    Bei,
    Ker,
    Kei,
    Spence,
    Zeta,
    Zetac,
    HurwitzZeta,
    Lpmv,
    SphHarmY,
    EvalLegendre,
    EvalChebyt,
    EvalChebyu,
    EvalHermite,
    EvalHermitenorm,
    EvalLaguerre,
    EvalGenlaguerre,
    EvalJacobi,
    EvalGegenbauer,
    EvalShLegendre,
    EvalShChebyt,
    EvalShChebyu,
    RootsChebytNode,
    RootsChebytWeight,
    RootsChebyuNode,
    RootsChebyuWeight,
    RootsHermiteNode,
    RootsHermiteWeight,
    RootsHermitenormNode,
    RootsHermitenormWeight,
    RootsLaguerreNode,
    RootsLaguerreWeight,
    RootsGenlaguerreNode,
    RootsGenlaguerreWeight,
    RootsJacobiNode,
    RootsJacobiWeight,
    RootsGegenbauerNode,
    RootsGegenbauerWeight,
    RootsLegendreNode,
    RootsLegendreWeight,
    Expit,
    Logit,
    Exprel,
    // ── Metamorphic-identity evaluators (strength-4 per /testing-fuzzing) ──
    //
    // Each variant below combines multiple fsci-special functions to
    // evaluate a mathematical identity (e.g. erf(x)+erfc(x)=1). The
    // comparator checks the result against the fixture-embedded expected
    // value.
    //
    // IMPORTANT: these are **self-consistency** oracles. They verify that
    // the Rust implementations agree WITH EACH OTHER. A bug that is
    // compensated between two identity legs (e.g., erf off by +ε and erfc
    // off by −ε) will still pass the identity check. Per bead
    // frankenscipy-tfd7: these are strength-4 metamorphic relations, NOT
    // strength-1 scipy-reference comparisons. They catch gross asymmetries
    // and sign flips but not shared arithmetic drift.
    //
    // For true scipy-parity verification of individual special functions,
    // use cases with category='differential' that dispatch to
    // scipy_special_oracle.py (per v2tz / ivg5 chains). The identity cases
    // below are complementary coverage, not a substitute.
    RelErfErfcIdentity,
    RelGammaRecurrence,
    RelBetaSymmetry,
    RelGammaincComplement,
    RelJnRecurrence,
    RelErfinvComposition,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpecialValueClass {
    Finite,
    Nan,
    PosInf,
    NegInf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpecialComplexValue {
    pub re: f64,
    pub im: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum SpecialArgument {
    RealScalar(f64),
    ComplexScalar(SpecialComplexValue),
    RealVector { values: Vec<f64> },
    ComplexVector { values: Vec<SpecialComplexValue> },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpecialExpectedOutcome {
    Scalar {
        value: f64,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    Vector {
        values: Vec<f64>,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    ComplexScalar {
        value: SpecialComplexValue,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    ComplexVector {
        values: Vec<SpecialComplexValue>,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    Class {
        class: SpecialValueClass,
    },
    ErrorKind {
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SpecialCase {
    pub case_id: String,
    pub category: String,
    pub mode: RuntimeMode,
    pub function: SpecialCaseFunction,
    #[serde(default)]
    pub args: Vec<SpecialArgument>,
    pub expected: SpecialExpectedOutcome,
}

impl SpecialCase {
    fn case_id(&self) -> &str {
        &self.case_id
    }

    fn real_scalar_args(&self) -> Option<Vec<f64>> {
        self.args
            .iter()
            .map(|arg| match arg {
                SpecialArgument::RealScalar(value) => Some(*value),
                _ => None,
            })
            .collect()
    }

    #[cfg(test)]
    fn category(&self) -> &str {
        &self.category
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SpecialPacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<SpecialCase>,
}

#[derive(Debug, Clone, PartialEq)]
enum SpecialObservedOutcome {
    Scalar(f64),
    Vector(Vec<f64>),
    ComplexScalar([f64; 2]),
    ComplexVector(Vec<[f64; 2]>),
}

// ---------- Sparse packet fixture types ----------

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SparseOperation {
    Spmv,
    Spsolve,
    FormatRoundtrip,
    Add,
    Scale,
    /// br-gorz: scipy.sparse.linalg.eigsh — symmetric/Hermitian
    /// largest-k eigenpairs.
    Eigsh,
    /// br-gorz: scipy.sparse.linalg.eigs — non-symmetric largest-k
    /// eigenpairs (real part of complex eigenvalues exposed).
    Eigs,
    /// br-gorz: scipy.sparse.linalg.svds — top-k singular values.
    Svds,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SparseInputFormat {
    Coo,
    Csr,
    Csc,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SparseExpectedOutcome {
    Vector {
        #[serde(
            deserialize_with = "deserialize_maybe_nan_required_vec",
            serialize_with = "serialize_maybe_nan_required_vec"
        )]
        values: Vec<f64>,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
    },
    Scalar {
        #[serde(with = "maybe_nan_f64")]
        value: f64,
        #[serde(default)]
        atol: Option<f64>,
    },
    Shape {
        rows: usize,
        cols: usize,
        nnz: usize,
    },
    Error {
        error: String,
    },
    /// br-gorz: top-k eigenvalues / singular values. Comparison is
    /// invariant to ordering: both observed and expected vectors are
    /// sorted by abs descending before allclose check.
    EigenvaluesAbsSorted {
        #[serde(
            deserialize_with = "deserialize_maybe_nan_required_vec",
            serialize_with = "serialize_maybe_nan_required_vec"
        )]
        values: Vec<f64>,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SparseCase {
    pub case_id: String,
    pub category: String,
    pub mode: RuntimeMode,
    pub operation: SparseOperation,
    pub format: SparseInputFormat,
    pub rows: usize,
    pub cols: usize,
    #[serde(
        deserialize_with = "deserialize_maybe_nan_required_vec",
        serialize_with = "serialize_maybe_nan_required_vec"
    )]
    pub data: Vec<f64>,
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    #[serde(
        default,
        deserialize_with = "deserialize_maybe_nan_vec",
        serialize_with = "serialize_maybe_nan_vec"
    )]
    pub rhs: Option<Vec<f64>>,
    #[serde(
        default,
        deserialize_with = "deserialize_maybe_nan_option_f64",
        serialize_with = "serialize_maybe_nan_option_f64"
    )]
    pub scalar: Option<f64>,
    /// br-gorz: number of eigenvalues / singular values to compute
    /// for eigsh/eigs/svds. Ignored by other operations.
    #[serde(default)]
    pub k: Option<usize>,
    pub expected: SparseExpectedOutcome,
}

impl SparseCase {
    fn case_id(&self) -> &str {
        &self.case_id
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SparsePacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<SparseCase>,
}

// ── FFT fixture types ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FftTransformKind {
    Fft,
    Ifft,
    Rfft,
    Irfft,
    Fft2,
    Ifft2,
    Fftn,
    Ifftn,
    Rfft2,
    Irfft2,
    Rfftn,
    Irfftn,
    Fftfreq,
    Rfftfreq,
    Fftshift,
    Ifftshift,
    Hfft,
    Ihfft,
    Dct,
    Idct,
    DctI,
    DctIii,
    DctIv,
    DstI,
    DstIi,
    DstIii,
    DstIv,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FftNormalization {
    Forward,
    Backward,
    Ortho,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FftExpectedOutcome {
    ComplexVector {
        values: Vec<[f64; 2]>,
        atol: Option<f64>,
        rtol: Option<f64>,
    },
    RealVector {
        values: Vec<f64>,
        atol: Option<f64>,
        rtol: Option<f64>,
    },
    Error {
        error: String,
    },
}

// br-gr99: NaN/Inf string-sentinel support for JSON fixtures.
//
// Stock JSON cannot encode NaN / +Inf / -Inf as number literals, but
// Hardened-mode conformance cases need non-finite inputs to exercise
// the reject paths (FftError::NonFiniteInput, LinalgError::NonFiniteInput,
// etc.). The deserializers below accept either a plain JSON number or
// one of the string markers "NaN" / "Infinity" / "-Infinity" (case-
// insensitive) and route both into f64. Round-trip is preserved
// because serialize_f64_with_nan_sentinel emits the same markers on
// write.
//
// Mirror this contract on the oracle side (scipy_*_oracle.py
// _coerce_maybe_nan_f64) so both sides see the same non-finite
// value for a given fixture entry.
mod maybe_nan_f64 {
    use serde::de::{self, Deserializer};
    use serde::ser::Serializer;

    pub(super) fn deserialize<'de, D>(de: D) -> Result<f64, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct V;
        impl<'de> de::Visitor<'de> for V {
            type Value = f64;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("f64 number or string sentinel NaN/Infinity/-Infinity")
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> Result<f64, E> {
                Ok(v)
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<f64, E> {
                Ok(v as f64)
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<f64, E> {
                Ok(v as f64)
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<f64, E> {
                match v.trim().to_ascii_lowercase().as_str() {
                    "nan" => Ok(f64::NAN),
                    "infinity" | "inf" | "+infinity" | "+inf" => Ok(f64::INFINITY),
                    "-infinity" | "-inf" => Ok(f64::NEG_INFINITY),
                    other => other.parse::<f64>().map_err(|_| {
                        de::Error::custom(format!("expected f64 or NaN/Inf sentinel, got {v:?}"))
                    }),
                }
            }
        }
        de.deserialize_any(V)
    }

    pub(super) fn serialize<S: Serializer>(value: &f64, s: S) -> Result<S::Ok, S::Error> {
        if value.is_nan() {
            s.serialize_str("NaN")
        } else if *value == f64::INFINITY {
            s.serialize_str("Infinity")
        } else if *value == f64::NEG_INFINITY {
            s.serialize_str("-Infinity")
        } else {
            s.serialize_f64(*value)
        }
    }
}

fn deserialize_maybe_nan_vec<'de, D>(de: D) -> Result<Option<Vec<f64>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(serde::Deserialize)]
    struct Wrap(#[serde(with = "maybe_nan_f64")] f64);
    let opt: Option<Vec<Wrap>> = Option::deserialize(de)?;
    Ok(opt.map(|v| v.into_iter().map(|Wrap(x)| x).collect()))
}

fn serialize_maybe_nan_vec<S: serde::Serializer>(
    value: &Option<Vec<f64>>,
    s: S,
) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;
    match value {
        None => s.serialize_none(),
        Some(v) => {
            let mut seq = s.serialize_seq(Some(v.len()))?;
            for x in v {
                seq.serialize_element(&MaybeNanF64(*x))?;
            }
            seq.end()
        }
    }
}

fn deserialize_maybe_nan_complex_vec<'de, D>(de: D) -> Result<Option<Vec<[f64; 2]>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(serde::Deserialize)]
    struct WrapPair(
        #[serde(with = "maybe_nan_f64")] f64,
        #[serde(with = "maybe_nan_f64")] f64,
    );
    let opt: Option<Vec<WrapPair>> = Option::deserialize(de)?;
    Ok(opt.map(|v| v.into_iter().map(|WrapPair(a, b)| [a, b]).collect()))
}

fn serialize_maybe_nan_complex_vec<S: serde::Serializer>(
    value: &Option<Vec<[f64; 2]>>,
    s: S,
) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;
    match value {
        None => s.serialize_none(),
        Some(v) => {
            let mut seq = s.serialize_seq(Some(v.len()))?;
            for pair in v {
                seq.serialize_element(&[MaybeNanF64(pair[0]), MaybeNanF64(pair[1])])?;
            }
            seq.end()
        }
    }
}

struct MaybeNanF64(f64);
impl serde::Serialize for MaybeNanF64 {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        maybe_nan_f64::serialize(&self.0, s)
    }
}

struct MaybeNanVec<'a>(&'a [f64]);

impl serde::Serialize for MaybeNanVec<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        serialize_maybe_nan_slice(self.0, s)
    }
}

fn serialize_maybe_nan_slice<S: serde::Serializer>(value: &[f64], s: S) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;

    let mut seq = s.serialize_seq(Some(value.len()))?;
    for x in value {
        seq.serialize_element(&MaybeNanF64(*x))?;
    }
    seq.end()
}

fn deserialize_maybe_nan_required_vec<'de, D>(de: D) -> Result<Vec<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(serde::Deserialize)]
    struct Wrap(#[serde(with = "maybe_nan_f64")] f64);

    let values: Vec<Wrap> = Vec::deserialize(de)?;
    Ok(values.into_iter().map(|Wrap(x)| x).collect())
}

fn serialize_maybe_nan_required_vec<S: serde::Serializer>(
    value: &[f64],
    s: S,
) -> Result<S::Ok, S::Error> {
    serialize_maybe_nan_slice(value, s)
}

fn deserialize_maybe_nan_matrix<'de, D>(de: D) -> Result<Vec<Vec<f64>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(serde::Deserialize)]
    struct Wrap(#[serde(with = "maybe_nan_f64")] f64);

    let rows: Vec<Vec<Wrap>> = Vec::deserialize(de)?;
    Ok(rows
        .into_iter()
        .map(|row| row.into_iter().map(|Wrap(x)| x).collect())
        .collect())
}

fn serialize_maybe_nan_matrix<S: serde::Serializer>(
    value: &[Vec<f64>],
    s: S,
) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;

    let mut seq = s.serialize_seq(Some(value.len()))?;
    for row in value {
        seq.serialize_element(&MaybeNanVec(row))?;
    }
    seq.end()
}

fn deserialize_maybe_nan_option_f64<'de, D>(de: D) -> Result<Option<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(serde::Deserialize)]
    struct Wrap(#[serde(with = "maybe_nan_f64")] f64);

    let value: Option<Wrap> = Option::deserialize(de)?;
    Ok(value.map(|Wrap(x)| x))
}

fn serialize_maybe_nan_option_f64<S: serde::Serializer>(
    value: &Option<f64>,
    s: S,
) -> Result<S::Ok, S::Error> {
    match value {
        None => s.serialize_none(),
        Some(value) => maybe_nan_f64::serialize(value, s),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FftCase {
    pub case_id: String,
    pub category: String,
    pub mode: RuntimeMode,
    pub transform: FftTransformKind,
    pub normalization: Option<FftNormalization>,
    /// Real input (for rfft, irfft output, fftfreq, rfftfreq, fftshift, ifftshift).
    /// Accepts NaN/Infinity/-Infinity string sentinels per br-gr99 so
    /// Hardened fixture cases can exercise non-finite reject paths.
    #[serde(
        default,
        deserialize_with = "deserialize_maybe_nan_vec",
        serialize_with = "serialize_maybe_nan_vec"
    )]
    pub real_input: Option<Vec<f64>>,
    /// Complex input as [[re, im], ...]. Each component accepts the
    /// same NaN/Infinity/-Infinity string sentinels per br-gr99.
    #[serde(
        default,
        deserialize_with = "deserialize_maybe_nan_complex_vec",
        serialize_with = "serialize_maybe_nan_complex_vec"
    )]
    pub complex_input: Option<Vec<[f64; 2]>>,
    /// For irfft: desired output length
    pub output_len: Option<usize>,
    /// For fftfreq/rfftfreq: sample spacing
    pub sample_spacing: Option<f64>,
    /// For 2D/ND transforms: shape
    pub shape: Option<Vec<usize>>,
    pub expected: FftExpectedOutcome,
}

impl FftCase {
    fn case_id(&self) -> &str {
        &self.case_id
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FftPacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<FftCase>,
}

// ── CASP Runtime fixture types ───────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CaspTestKind {
    PolicyDecision,
    SolverSelection,
    CalibratorDrift,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CaspExpectedOutcome {
    PolicyAction { action: String },
    SolverAction { action: String },
    CalibratorFallback { should_fallback: bool },
    Error { error: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CaspCase {
    pub case_id: String,
    pub category: String,
    pub mode: RuntimeMode,
    pub test_kind: CaspTestKind,
    /// For policy decisions: condition_signal, metadata_signal, anomaly_signal
    pub condition_signal: Option<f64>,
    pub metadata_signal: Option<f64>,
    pub anomaly_signal: Option<f64>,
    /// For solver selection: matrix condition state
    pub condition_state: Option<String>,
    /// For calibrator: sequence of backward error observations
    pub observations: Option<Vec<f64>>,
    /// PolicyController/SolverPortfolio ring-buffer capacity. Defaults to
    /// 64 for backwards compatibility with existing fixtures. Per
    /// frankenscipy-p71m: previously hardcoded in execute_casp_case.
    #[serde(default)]
    pub capacity: Option<usize>,
    /// ConformalCalibrator significance level (alpha). Defaults to 0.05.
    /// Per frankenscipy-p71m.
    #[serde(default)]
    pub alpha: Option<f64>,
    /// ConformalCalibrator history capacity. Defaults to 200 per the
    /// calibrator convention. Per frankenscipy-p71m.
    #[serde(default)]
    pub calibrator_capacity: Option<usize>,
    pub expected: CaspExpectedOutcome,
}

impl CaspCase {
    fn case_id(&self) -> &str {
        &self.case_id
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CaspPacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<CaspCase>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArrayApiFixtureDType {
    Bool,
    Int64,
    UInt64,
    Float32,
    Float64,
    Complex64,
    Complex128,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ArrayApiFixtureScalar {
    Bool { value: bool },
    I64 { value: i64 },
    U64 { value: u64 },
    F64 { value: f64 },
    ComplexF64 { re: f64, im: f64 },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArrayApiFixtureIndexingMode {
    Basic,
    Advanced,
    BooleanMask,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArrayApiFixtureSliceSpec {
    pub start: Option<isize>,
    pub stop: Option<isize>,
    pub step: isize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ArrayApiFixtureIndexExpr {
    Basic {
        slices: Vec<ArrayApiFixtureSliceSpec>,
    },
    Advanced {
        indices: Vec<Vec<isize>>,
    },
    BooleanMask {
        mask_shape: Vec<usize>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ArrayApiExpectedOutcome {
    Array {
        shape: Vec<usize>,
        dtype: ArrayApiFixtureDType,
        values: Vec<ArrayApiFixtureScalar>,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    Shape {
        dims: Vec<usize>,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    Dtype {
        dtype: ArrayApiFixtureDType,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    Bool {
        value: bool,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    ErrorKind {
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum ArrayApiCase {
    Zeros {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        shape: Vec<usize>,
        dtype: ArrayApiFixtureDType,
        expected: ArrayApiExpectedOutcome,
    },
    Ones {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        shape: Vec<usize>,
        dtype: ArrayApiFixtureDType,
        expected: ArrayApiExpectedOutcome,
    },
    Full {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        shape: Vec<usize>,
        fill_value: ArrayApiFixtureScalar,
        dtype: ArrayApiFixtureDType,
        expected: ArrayApiExpectedOutcome,
    },
    Arange {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        start: ArrayApiFixtureScalar,
        stop: ArrayApiFixtureScalar,
        step: ArrayApiFixtureScalar,
        #[serde(default)]
        dtype: Option<ArrayApiFixtureDType>,
        expected: ArrayApiExpectedOutcome,
    },
    Linspace {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        start: ArrayApiFixtureScalar,
        stop: ArrayApiFixtureScalar,
        num: usize,
        endpoint: bool,
        #[serde(default)]
        dtype: Option<ArrayApiFixtureDType>,
        expected: ArrayApiExpectedOutcome,
    },
    BroadcastShapes {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        shapes: Vec<Vec<usize>>,
        expected: ArrayApiExpectedOutcome,
    },
    ResultType {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        dtypes: Vec<ArrayApiFixtureDType>,
        #[serde(default)]
        force_floating: bool,
        expected: ArrayApiExpectedOutcome,
    },
    FromSlice {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        values: Vec<ArrayApiFixtureScalar>,
        shape: Vec<usize>,
        dtype: ArrayApiFixtureDType,
        expected: ArrayApiExpectedOutcome,
    },
    Getitem {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        source_values: Vec<ArrayApiFixtureScalar>,
        source_shape: Vec<usize>,
        source_dtype: ArrayApiFixtureDType,
        indexing_mode: ArrayApiFixtureIndexingMode,
        index: ArrayApiFixtureIndexExpr,
        expected: ArrayApiExpectedOutcome,
    },
    Reshape {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        source_values: Vec<ArrayApiFixtureScalar>,
        source_shape: Vec<usize>,
        source_dtype: ArrayApiFixtureDType,
        new_shape: Vec<usize>,
        expected: ArrayApiExpectedOutcome,
    },
    Transpose {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        source_values: Vec<ArrayApiFixtureScalar>,
        source_shape: Vec<usize>,
        source_dtype: ArrayApiFixtureDType,
        expected: ArrayApiExpectedOutcome,
    },
    RelationBroadcastCommutative {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        left_shape: Vec<usize>,
        right_shape: Vec<usize>,
        expected: ArrayApiExpectedOutcome,
    },
    RelationResultTypeSymmetry {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        left_dtype: ArrayApiFixtureDType,
        right_dtype: ArrayApiFixtureDType,
        #[serde(default)]
        force_floating: bool,
        expected: ArrayApiExpectedOutcome,
    },
    RelationIndexRoundtrip {
        case_id: String,
        category: String,
        mode: RuntimeMode,
        values: Vec<ArrayApiFixtureScalar>,
        dtype: ArrayApiFixtureDType,
        index: isize,
        expected: ArrayApiExpectedOutcome,
    },
}

impl ArrayApiCase {
    fn case_id(&self) -> &str {
        match self {
            Self::Zeros { case_id, .. }
            | Self::Ones { case_id, .. }
            | Self::Full { case_id, .. }
            | Self::Arange { case_id, .. }
            | Self::Linspace { case_id, .. }
            | Self::BroadcastShapes { case_id, .. }
            | Self::ResultType { case_id, .. }
            | Self::FromSlice { case_id, .. }
            | Self::Getitem { case_id, .. }
            | Self::Reshape { case_id, .. }
            | Self::Transpose { case_id, .. }
            | Self::RelationBroadcastCommutative { case_id, .. }
            | Self::RelationResultTypeSymmetry { case_id, .. }
            | Self::RelationIndexRoundtrip { case_id, .. } => case_id,
        }
    }

    #[cfg(test)]
    fn category(&self) -> &str {
        match self {
            Self::Zeros { category, .. }
            | Self::Ones { category, .. }
            | Self::Full { category, .. }
            | Self::Arange { category, .. }
            | Self::Linspace { category, .. }
            | Self::BroadcastShapes { category, .. }
            | Self::ResultType { category, .. }
            | Self::FromSlice { category, .. }
            | Self::Getitem { category, .. }
            | Self::Reshape { category, .. }
            | Self::Transpose { category, .. }
            | Self::RelationBroadcastCommutative { category, .. }
            | Self::RelationResultTypeSymmetry { category, .. }
            | Self::RelationIndexRoundtrip { category, .. } => category,
        }
    }

    fn expected(&self) -> &ArrayApiExpectedOutcome {
        match self {
            Self::Zeros { expected, .. }
            | Self::Ones { expected, .. }
            | Self::Full { expected, .. }
            | Self::Arange { expected, .. }
            | Self::Linspace { expected, .. }
            | Self::BroadcastShapes { expected, .. }
            | Self::ResultType { expected, .. }
            | Self::FromSlice { expected, .. }
            | Self::Getitem { expected, .. }
            | Self::Reshape { expected, .. }
            | Self::Transpose { expected, .. }
            | Self::RelationBroadcastCommutative { expected, .. }
            | Self::RelationResultTypeSymmetry { expected, .. }
            | Self::RelationIndexRoundtrip { expected, .. } => expected,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ArrayApiPacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<ArrayApiCase>,
}

#[derive(Debug, Clone, PartialEq)]
enum ArrayApiObservedOutcome {
    Array {
        shape: Vec<usize>,
        dtype: ArrayApiFixtureDType,
        values: Vec<ArrayApiFixtureScalar>,
    },
    Shape {
        dims: Vec<usize>,
    },
    Dtype {
        dtype: ArrayApiFixtureDType,
    },
    Bool {
        value: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaseResult {
    pub case_id: String,
    pub passed: bool,
    pub message: String,
}

const fn packet_report_schema_v1() -> u8 {
    1
}

const fn packet_report_schema_v2() -> u8 {
    2
}

/// How a `PacketReport` was produced, recorded in the emitted JSON so
/// downstream dashboard / audit consumers can distinguish self-check from
/// oracle-backed reports without heuristic filesystem probing. Per
/// frankenscipy-fytm: evidence_p2c*.rs tests were writing
/// `parity_report.json` with Rust-only self-check identities; the name
/// suggests scipy-parity verification but the content was self-consistent.
/// The `report_kind` field (skipped when serializing the default
/// `Unspecified`) now makes the distinction explicit in the JSON.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReportKind {
    /// Legacy/unknown origin. Present when an old writer didn't set the
    /// field; serializers skip it. Treat as 'unknown' — do NOT assume
    /// scipy-backed.
    #[default]
    Unspecified,
    /// Report generated from a scipy-backed differential comparison
    /// (run_*_packet_with_oracle_capture or run_differential_* with
    /// oracle_status.available).
    OracleBacked,
    /// Report generated from Rust-only self-check identities and/or
    /// fixture-embedded expected values. No scipy consulted. Includes
    /// the evidence_p2c*.rs final-pack outputs.
    SelfCheck,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PacketReport {
    #[serde(default = "packet_report_schema_v1")]
    pub schema_version: u8,
    pub packet_id: String,
    pub family: String,
    pub case_results: Vec<CaseResult>,
    pub passed_cases: usize,
    pub failed_cases: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixture_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oracle_status: Option<OracleStatus>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub differential_case_results: Option<Vec<DifferentialCaseResult>>,
    /// Classification of how the report was produced. Defaults to
    /// `Unspecified` for backwards compatibility — evidence_p2c*.rs
    /// tests should set this to `SelfCheck` per frankenscipy-fytm.
    #[serde(default, skip_serializing_if = "is_unspecified_report_kind")]
    pub report_kind: ReportKind,
    pub generated_unix_ms: u128,
}

fn is_unspecified_report_kind(kind: &ReportKind) -> bool {
    matches!(kind, ReportKind::Unspecified)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PacketSummary {
    pub packet_id: String,
    pub family: String,
    pub passed_cases: usize,
    pub failed_cases: usize,
    pub total_cases: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RaptorQSidecar {
    pub schema_version: u8,
    pub source_hash: String,
    pub symbol_size: usize,
    #[serde(default)]
    pub seed: u64,
    pub source_symbols: usize,
    pub repair_symbols: usize,
    pub repair_symbol_hashes: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub repair_symbol_payloads_hex: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecodeProofArtifact {
    pub ts_unix_ms: u128,
    pub reason: String,
    pub recovered_blocks: usize,
    pub proof_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParityArtifactBundle {
    pub report_path: PathBuf,
    pub sidecar_path: PathBuf,
    pub decode_proof_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OracleCapture {
    pub packet_id: String,
    pub family: String,
    pub generated_unix_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<OracleRuntimeInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<OracleCaptureProvenance>,
    pub case_outputs: Vec<OracleCaseOutput>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OracleRuntimeInfo {
    pub python_version: String,
    pub numpy_version: String,
    pub scipy_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OracleCaptureProvenance {
    pub fixture_input_blake3: String,
    pub oracle_output_blake3: String,
    pub capture_blake3: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OracleCaseOutput {
    pub case_id: String,
    pub status: String,
    pub result_kind: String,
    pub result: serde_json::Value,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PythonOracleConfig {
    pub python_bin: PathBuf,
    pub script_path: PathBuf,
    pub required: bool,
}

impl Default for PythonOracleConfig {
    fn default() -> Self {
        // Sentinel script path that does not exist; callers MUST either
        // route via `resolve_differential_oracle_config(family)` or set
        // `script_path` explicitly. This avoids the latent misrouting
        // where a non-linalg family silently invoked scipy_linalg_oracle.py
        // because the old default pointed there (bpmm).
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            python_bin: PathBuf::from("python3"),
            script_path: manifest.join("python_oracle/NO_DEFAULT_ORACLE_SET.py"),
            required: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum LinalgObservedOutcome {
    Vector {
        values: Vec<f64>,
        warning_ill_conditioned: bool,
    },
    Matrix {
        values: Vec<Vec<f64>>,
    },
    Scalar {
        value: f64,
    },
    Lstsq {
        x: Vec<f64>,
        residuals: Vec<f64>,
        rank: usize,
        singular_values: Vec<f64>,
    },
    Pinv {
        values: Vec<Vec<f64>>,
        rank: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
enum OptimizeObservedOutcome {
    Minimize {
        x: Vec<f64>,
        fun: Option<f64>,
        success: bool,
        status: OptConvergenceStatus,
    },
    Root {
        root: f64,
        converged: bool,
        status: OptConvergenceStatus,
    },
}

#[derive(Debug, Clone, PartialEq)]
enum InterpolateObservedOutcome {
    Vector(Vec<f64>),
    Error(String),
}

#[derive(Debug, Clone, PartialEq)]
enum IoObservedOutcome {
    Matrix {
        rows: usize,
        cols: usize,
        values: Vec<f64>,
    },
    Wav {
        sample_rate: u32,
        channels: u16,
        bits_per_sample: u16,
        values: Vec<f64>,
    },
    Error(String),
}

#[derive(Debug, Clone, PartialEq)]
enum NdimageObservedOutcome {
    Array {
        values: Vec<f64>,
        shape: Vec<usize>,
    },
    Label {
        labels: Vec<f64>,
        shape: Vec<usize>,
        num_features: usize,
    },
    Error(String),
}

pub fn run_smoke(config: &HarnessConfig) -> Result<HarnessReport, HarnessError> {
    let packet = run_validate_tol_packet(config, "FSCI-P2C-001_validate_tol.json")?;
    Ok(HarnessReport {
        suite: "smoke",
        oracle_present: config.oracle_root.exists(),
        cases_run: packet.passed_cases + packet.failed_cases,
        failed_cases: packet.failed_cases,
        strict_mode: config.strict_mode,
    })
}

pub fn run_validate_tol_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: PacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let outcome = validate_tol(
            case.rtol.clone().into(),
            case.atol.clone().into(),
            case.n,
            case.mode,
        );

        let (passed, message) = match (&case.expected, outcome) {
            (
                ExpectedOutcome::Ok {
                    rtol,
                    atol,
                    warning_rtol_clamped,
                },
                Ok(actual),
            ) => {
                let actual_warning = !actual.warnings.is_empty();
                let expected_rtol: ToleranceValue = rtol.clone().into();
                let expected_atol: ToleranceValue = atol.clone().into();
                let pass = actual.rtol == expected_rtol
                    && actual.atol == expected_atol
                    && actual_warning == *warning_rtol_clamped;
                let msg = if pass {
                    "output matched expected tolerance contract".to_owned()
                } else {
                    format!(
                        "mismatch: expected rtol={expected_rtol:?}, atol={expected_atol:?}, warning={warning_rtol_clamped}; got rtol={:?}, atol={:?}, warning={actual_warning}",
                        actual.rtol, actual.atol
                    )
                };
                (pass, msg)
            }
            (ExpectedOutcome::Error { error }, Err(actual)) => {
                let pass = matches_error_contract(&actual.to_string(), error);
                let msg = if pass {
                    "error matched expected contract".to_owned()
                } else {
                    format!("mismatch: expected error `{error}`, got `{actual}`")
                };
                (pass, msg)
            }
            (expected, result) => (
                false,
                format!("shape mismatch: expected {expected:?}, got {result:?}"),
            ),
        };

        case_results.push(CaseResult {
            case_id: case.case_id.clone(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

pub fn run_linalg_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: LinalgPacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute_linalg_case(case)
        })) {
            Ok(v) => v,
            Err(payload) => {
                case_results.push(CaseResult {
                    case_id: case.case_id().to_owned(),
                    passed: false,
                    message: format!(
                        "PANIC in execute_linalg_case: {}",
                        panic_payload_message(payload)
                    ),
                });
                continue;
            }
        };
        let (passed, message) = compare_linalg_case(case.expected(), &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

fn run_linalg_packet_against_oracle_capture(
    config: &HarnessConfig,
    fixture_name: &str,
    oracle_capture: &OracleCapture,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: LinalgPacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    if oracle_capture.packet_id != fixture.packet_id {
        return Err(HarnessError::OracleCaptureMismatch {
            detail: format!(
                "packet id mismatch: fixture={} oracle={}",
                fixture.packet_id, oracle_capture.packet_id
            ),
        });
    }

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute_linalg_case(case)
        })) {
            Ok(v) => v,
            Err(payload) => {
                case_results.push(CaseResult {
                    case_id: case.case_id().to_owned(),
                    passed: false,
                    message: format!(
                        "PANIC in execute_linalg_case: {}",
                        panic_payload_message(payload)
                    ),
                });
                continue;
            }
        };
        let Some(oracle_case) = oracle_capture
            .case_outputs
            .iter()
            .find(|entry| entry.case_id == case.case_id())
        else {
            case_results.push(CaseResult {
                case_id: case.case_id().to_owned(),
                passed: false,
                message: "missing oracle capture output".to_owned(),
            });
            continue;
        };

        let (passed, message, _, _) =
            compare_linalg_case_against_oracle(case, oracle_case, &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

pub fn run_optimize_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: OptimizePacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute_optimize_case(case)
        })) {
            Ok(v) => v,
            Err(payload) => {
                case_results.push(CaseResult {
                    case_id: case.case_id().to_owned(),
                    passed: false,
                    message: format!(
                        "PANIC in execute_optimize_case: {}",
                        panic_payload_message(payload)
                    ),
                });
                continue;
            }
        };
        let (passed, message, _, _) =
            compare_optimize_case_differential(case.expected(), &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

pub fn run_interpolate_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: InterpolatePacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute_interpolate_case(case)
        })) {
            Ok(v) => v,
            Err(payload) => {
                case_results.push(CaseResult {
                    case_id: case.case_id().to_owned(),
                    passed: false,
                    message: format!(
                        "PANIC in execute_interpolate_case: {}",
                        panic_payload_message(payload)
                    ),
                });
                continue;
            }
        };
        let (passed, message, _, _) = compare_interpolate_case_differential(case, &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

pub fn run_io_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: IoPacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute_io_case(case)
        })) {
            Ok(v) => v,
            Err(payload) => {
                case_results.push(CaseResult {
                    case_id: case.case_id().to_owned(),
                    passed: false,
                    message: format!(
                        "PANIC in execute_io_case: {}",
                        panic_payload_message(payload)
                    ),
                });
                continue;
            }
        };
        let (passed, message, _, _) = compare_io_case_differential(case, &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

pub fn run_constants_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let oracle_config = DifferentialOracleConfig::default();
    let report = run_differential_test(&fixture_path, &oracle_config)?;
    Ok(PacketReport::from(&report))
}

pub fn run_ndimage_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: NdimagePacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute_ndimage_case(case)
        })) {
            Ok(v) => v,
            Err(payload) => {
                case_results.push(CaseResult {
                    case_id: case.case_id().to_owned(),
                    passed: false,
                    message: format!(
                        "PANIC in execute_ndimage_case: {}",
                        panic_payload_message(payload)
                    ),
                });
                continue;
            }
        };
        let (passed, message, _, _) = compare_ndimage_case_differential(case, &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

pub fn run_special_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: SpecialPacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute_special_case(case)
        })) {
            Ok(v) => v,
            Err(payload) => {
                case_results.push(CaseResult {
                    case_id: case.case_id().to_owned(),
                    passed: false,
                    message: format!(
                        "PANIC in execute_special_case: {}",
                        panic_payload_message(payload)
                    ),
                });
                continue;
            }
        };
        let (passed, message, _, _) = compare_special_case_differential(case, &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

pub fn run_array_api_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: ArrayApiPacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute_array_api_case(case)
        })) {
            Ok(v) => v,
            Err(payload) => {
                case_results.push(CaseResult {
                    case_id: case.case_id().to_owned(),
                    passed: false,
                    message: format!(
                        "PANIC in execute_array_api_case: {}",
                        panic_payload_message(payload)
                    ),
                });
                continue;
            }
        };
        let (passed, message, _, _) = compare_array_api_case_differential(case, &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

pub fn run_sparse_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: SparsePacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute_sparse_case(case)
        })) {
            Ok(v) => v,
            Err(payload) => {
                case_results.push(CaseResult {
                    case_id: case.case_id().to_owned(),
                    passed: false,
                    message: format!(
                        "PANIC in execute_sparse_case: {}",
                        panic_payload_message(payload)
                    ),
                });
                continue;
            }
        };
        let (passed, message) = compare_sparse_outcome(&case.expected, &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

#[derive(Debug)]
enum SparseObserved {
    Vector(Vec<f64>),
    Shape {
        rows: usize,
        cols: usize,
        nnz: usize,
    },
    /// br-gorz: top-k eigenvalues or singular values. Sorted by abs
    /// descending in the comparator.
    EigenvaluesAbsSorted(Vec<f64>),
    Error(String),
}

fn execute_sparse_case(case: &SparseCase) -> SparseObserved {
    use fsci_sparse::{CooMatrix, FormatConvertible, Shape2D, SolveOptions};

    let shape = Shape2D::new(case.rows, case.cols);

    let coo = match CooMatrix::from_triplets(
        shape,
        case.data.clone(),
        case.row_indices.clone(),
        case.col_indices.clone(),
        true,
    ) {
        Ok(m) => m,
        Err(e) => return SparseObserved::Error(format!("{e}")),
    };

    match case.operation {
        SparseOperation::Spmv => {
            let rhs = match &case.rhs {
                Some(v) => v.as_slice(),
                None => return SparseObserved::Error("spmv requires rhs".to_owned()),
            };
            match case.format {
                SparseInputFormat::Coo => match fsci_sparse::spmv_coo(&coo, rhs) {
                    Ok(v) => SparseObserved::Vector(v),
                    Err(e) => SparseObserved::Error(format!("{e}")),
                },
                SparseInputFormat::Csr => {
                    let csr = match coo.to_csr() {
                        Ok(c) => c,
                        Err(e) => return SparseObserved::Error(format!("{e}")),
                    };
                    match fsci_sparse::spmv_csr(&csr, rhs) {
                        Ok(v) => SparseObserved::Vector(v),
                        Err(e) => SparseObserved::Error(format!("{e}")),
                    }
                }
                SparseInputFormat::Csc => {
                    let csc = match coo.to_csc() {
                        Ok(c) => c,
                        Err(e) => return SparseObserved::Error(format!("{e}")),
                    };
                    match fsci_sparse::spmv_csc(&csc, rhs) {
                        Ok(v) => SparseObserved::Vector(v),
                        Err(e) => SparseObserved::Error(format!("{e}")),
                    }
                }
            }
        }
        SparseOperation::Spsolve => {
            let rhs = match &case.rhs {
                Some(v) => v.as_slice(),
                None => return SparseObserved::Error("spsolve requires rhs".to_owned()),
            };
            let csr = match coo.to_csr() {
                Ok(c) => c,
                Err(e) => return SparseObserved::Error(format!("{e}")),
            };
            // Use case.mode to respect Hardened vs Strict from fixture
            let options = SolveOptions {
                mode: case.mode,
                ..SolveOptions::default()
            };
            match fsci_sparse::spsolve(&csr, rhs, options) {
                Ok(result) => SparseObserved::Vector(result.solution),
                Err(e) => SparseObserved::Error(format!("{e}")),
            }
        }
        SparseOperation::FormatRoundtrip => {
            let csr = match coo.to_csr() {
                Ok(c) => c,
                Err(e) => return SparseObserved::Error(format!("{e}")),
            };
            let csc = match csr.to_csc() {
                Ok(c) => c,
                Err(e) => return SparseObserved::Error(format!("{e}")),
            };
            // Verify CSC→CSR roundtrip preserves structure
            let csr2 = match csc.to_csr() {
                Ok(c) => c,
                Err(e) => return SparseObserved::Error(format!("{e}")),
            };
            SparseObserved::Shape {
                rows: csr2.shape().rows,
                cols: csr2.shape().cols,
                nnz: csr2.nnz(),
            }
        }
        SparseOperation::Add => {
            let csr_a = match coo.to_csr() {
                Ok(c) => c,
                Err(e) => return SparseObserved::Error(format!("{e}")),
            };
            match fsci_sparse::add_csr(&csr_a, &csr_a) {
                Ok(result) => {
                    let rhs = match &case.rhs {
                        Some(v) => v.as_slice(),
                        None => {
                            return SparseObserved::Shape {
                                rows: result.shape().rows,
                                cols: result.shape().cols,
                                nnz: result.nnz(),
                            };
                        }
                    };
                    match fsci_sparse::spmv_csr(&result, rhs) {
                        Ok(v) => SparseObserved::Vector(v),
                        Err(e) => SparseObserved::Error(format!("{e}")),
                    }
                }
                Err(e) => SparseObserved::Error(format!("{e}")),
            }
        }
        SparseOperation::Scale => {
            let csr = match coo.to_csr() {
                Ok(c) => c,
                Err(e) => return SparseObserved::Error(format!("{e}")),
            };
            let scalar = case.scalar.unwrap_or(1.0);
            match fsci_sparse::scale_csr(&csr, scalar) {
                Ok(result) => {
                    let rhs = match &case.rhs {
                        Some(v) => v.as_slice(),
                        None => {
                            return SparseObserved::Shape {
                                rows: result.shape().rows,
                                cols: result.shape().cols,
                                nnz: result.nnz(),
                            };
                        }
                    };
                    match fsci_sparse::spmv_csr(&result, rhs) {
                        Ok(v) => SparseObserved::Vector(v),
                        Err(e) => SparseObserved::Error(format!("{e}")),
                    }
                }
                Err(e) => SparseObserved::Error(format!("{e}")),
            }
        }
        SparseOperation::Eigsh => {
            // br-gorz: symmetric/Hermitian top-k eigenvalues.
            let csr = match coo.to_csr() {
                Ok(c) => c,
                Err(e) => return SparseObserved::Error(format!("{e}")),
            };
            let k = case.k.unwrap_or(1);
            match fsci_sparse::eigsh(&csr, k, fsci_sparse::EigsOptions::default()) {
                Ok(res) => SparseObserved::EigenvaluesAbsSorted(res.eigenvalues),
                Err(e) => SparseObserved::Error(format!("{e}")),
            }
        }
        SparseOperation::Eigs => {
            let csr = match coo.to_csr() {
                Ok(c) => c,
                Err(e) => return SparseObserved::Error(format!("{e}")),
            };
            let k = case.k.unwrap_or(1);
            match fsci_sparse::eigs(&csr, k, fsci_sparse::EigsOptions::default()) {
                Ok(res) => SparseObserved::EigenvaluesAbsSorted(res.eigenvalues),
                Err(e) => SparseObserved::Error(format!("{e}")),
            }
        }
        SparseOperation::Svds => {
            let csr = match coo.to_csr() {
                Ok(c) => c,
                Err(e) => return SparseObserved::Error(format!("{e}")),
            };
            let k = case.k.unwrap_or(1);
            match fsci_sparse::svds(&csr, k, fsci_sparse::EigsOptions::default()) {
                Ok(res) => SparseObserved::EigenvaluesAbsSorted(res.singular_values),
                Err(e) => SparseObserved::Error(format!("{e}")),
            }
        }
    }
}

/// br-gorz: sort by |x| descending so eigenvalue/singular-value order
/// is normalized between scipy and fsci before allclose comparison.
fn sort_abs_desc(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| b.abs().total_cmp(&a.abs()));
    v
}

fn compare_sparse_outcome(
    expected: &SparseExpectedOutcome,
    observed: &SparseObserved,
) -> (bool, String) {
    match (expected, observed) {
        (SparseExpectedOutcome::Vector { values, atol, rtol }, SparseObserved::Vector(got)) => {
            let a = atol.unwrap_or(1.0e-10);
            let r = rtol.unwrap_or(1.0e-8);
            if got.len() != values.len() {
                return (
                    false,
                    format!(
                        "vector length mismatch: expected {} got {}",
                        values.len(),
                        got.len()
                    ),
                );
            }
            let max_diff = got
                .iter()
                .zip(values.iter())
                .map(|(g, e)| (g - e).abs())
                .fold(0.0_f64, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                });
            let pass = got
                .iter()
                .zip(values.iter())
                .all(|(g, e)| allclose_scalar(*g, *e, a, r));
            if pass {
                (true, format!("vector matched (max_diff={max_diff:.2e})"))
            } else {
                (
                    false,
                    format!("vector mismatch: max_diff={max_diff:.2e}, atol={a:.2e}, rtol={r:.2e}"),
                )
            }
        }
        (
            SparseExpectedOutcome::Shape { rows, cols, nnz },
            SparseObserved::Shape {
                rows: gr,
                cols: gc,
                nnz: gn,
            },
        ) => {
            let pass = *rows == *gr && *cols == *gc && *nnz == *gn;
            if pass {
                (true, format!("shape matched ({gr}x{gc}, nnz={gn})"))
            } else {
                (
                    false,
                    format!(
                        "shape mismatch: expected {rows}x{cols} nnz={nnz}, got {gr}x{gc} nnz={gn}"
                    ),
                )
            }
        }
        (SparseExpectedOutcome::Error { error }, SparseObserved::Error(got)) => {
            let pass = got.contains(error.as_str());
            if pass {
                (true, format!("error matched: {got}"))
            } else {
                (
                    false,
                    format!("error mismatch: expected '{error}', got '{got}'"),
                )
            }
        }
        (SparseExpectedOutcome::Error { error }, _) => {
            (false, format!("expected error '{error}' but got success"))
        }
        (
            SparseExpectedOutcome::EigenvaluesAbsSorted {
                values: expected_values,
                atol,
                rtol,
            },
            SparseObserved::EigenvaluesAbsSorted(got),
        ) => {
            // br-gorz: order-invariant compare. Sort both by |x| desc.
            if expected_values.len() != got.len() {
                return (
                    false,
                    format!(
                        "eigenvalue count mismatch: expected {}, got {}",
                        expected_values.len(),
                        got.len()
                    ),
                );
            }
            let exp_sorted = sort_abs_desc(expected_values.clone());
            let got_sorted = sort_abs_desc(got.clone());
            let atol = atol.unwrap_or(1e-7);
            let rtol = rtol.unwrap_or(1e-7);
            for (i, (e, g)) in exp_sorted.iter().zip(got_sorted.iter()).enumerate() {
                if !allclose_scalar(*g, *e, atol, rtol) {
                    return (
                        false,
                        format!(
                            "eigenvalue[{i}] mismatch: expected {e}, got {g}, |diff|={:.3e}",
                            (e - g).abs()
                        ),
                    );
                }
            }
            (
                true,
                format!("eigenvalues matched ({} values, abs-sorted)", got.len()),
            )
        }
        (_, SparseObserved::Error(e)) => (false, format!("unexpected error: {e}")),
        _ => (false, "outcome type mismatch".to_owned()),
    }
}

// ── FFT packet runner ────────────────────────────────────────────────

pub fn run_fft_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: FftPacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed =
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute_fft_case(case)))
            {
                Ok(v) => v,
                Err(payload) => {
                    case_results.push(CaseResult {
                        case_id: case.case_id().to_owned(),
                        passed: false,
                        message: format!(
                            "PANIC in execute_fft_case: {}",
                            panic_payload_message(payload)
                        ),
                    });
                    continue;
                }
            };
        let (passed, message) = compare_fft_outcome(&case.expected, &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

#[derive(Debug)]
enum FftObserved {
    ComplexVector(Vec<[f64; 2]>),
    RealVector(Vec<f64>),
    Error(String),
}

fn execute_fft_case(case: &FftCase) -> FftObserved {
    use fsci_fft::{FftOptions, Normalization};

    let mut opts = FftOptions::default().with_mode(case.mode);
    if let Some(norm) = &case.normalization {
        opts = opts.with_normalization(match norm {
            FftNormalization::Forward => Normalization::Forward,
            FftNormalization::Backward => Normalization::Backward,
            FftNormalization::Ortho => Normalization::Ortho,
        });
    }

    match case.transform {
        FftTransformKind::Fft => {
            let input: Vec<(f64, f64)> = match &case.complex_input {
                Some(v) => v.iter().map(|c| (c[0], c[1])).collect(),
                None => return FftObserved::Error("fft requires complex_input".to_owned()),
            };
            match fsci_fft::fft(&input, &opts) {
                Ok(v) => FftObserved::ComplexVector(v.iter().map(|c| [c.0, c.1]).collect()),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Ifft => {
            let input: Vec<(f64, f64)> = match &case.complex_input {
                Some(v) => v.iter().map(|c| (c[0], c[1])).collect(),
                None => return FftObserved::Error("ifft requires complex_input".to_owned()),
            };
            match fsci_fft::ifft(&input, &opts) {
                Ok(v) => FftObserved::ComplexVector(v.iter().map(|c| [c.0, c.1]).collect()),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Rfft => {
            let input = match &case.real_input {
                Some(v) => v.as_slice(),
                None => return FftObserved::Error("rfft requires real_input".to_owned()),
            };
            match fsci_fft::rfft(input, &opts) {
                Ok(v) => FftObserved::ComplexVector(v.iter().map(|c| [c.0, c.1]).collect()),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Irfft => {
            let input: Vec<(f64, f64)> = match &case.complex_input {
                Some(v) => v.iter().map(|c| (c[0], c[1])).collect(),
                None => return FftObserved::Error("irfft requires complex_input".to_owned()),
            };
            match fsci_fft::irfft(&input, case.output_len, &opts) {
                Ok(v) => FftObserved::RealVector(v),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Rfft2 => {
            let input = match &case.real_input {
                Some(v) => v.as_slice(),
                None => return FftObserved::Error("rfft2 requires real_input".to_owned()),
            };
            let shape = match &case.shape {
                Some(s) => s.clone(),
                None => return FftObserved::Error("rfft2 requires shape".to_owned()),
            };
            if shape.len() != 2 {
                return FftObserved::Error("rfft2 requires 2D shape".to_owned());
            }
            match fsci_fft::rfft2(input, (shape[0], shape[1]), &opts) {
                Ok(v) => FftObserved::ComplexVector(v.iter().map(|c| [c.0, c.1]).collect()),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Irfft2 => {
            let input: Vec<(f64, f64)> = match &case.complex_input {
                Some(v) => v.iter().map(|c| (c[0], c[1])).collect(),
                None => return FftObserved::Error("irfft2 requires complex_input".to_owned()),
            };
            let shape = match &case.shape {
                Some(s) => s.clone(),
                None => return FftObserved::Error("irfft2 requires shape".to_owned()),
            };
            if shape.len() != 2 {
                return FftObserved::Error("irfft2 requires 2D shape".to_owned());
            }
            match fsci_fft::irfft2(&input, (shape[0], shape[1]), &opts) {
                Ok(v) => FftObserved::RealVector(v),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Rfftn => {
            let input = match &case.real_input {
                Some(v) => v.as_slice(),
                None => return FftObserved::Error("rfftn requires real_input".to_owned()),
            };
            let shape = match &case.shape {
                Some(s) => s.clone(),
                None => return FftObserved::Error("rfftn requires shape".to_owned()),
            };
            match fsci_fft::rfftn(input, &shape, &opts) {
                Ok(v) => FftObserved::ComplexVector(v.iter().map(|c| [c.0, c.1]).collect()),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Irfftn => {
            let input: Vec<(f64, f64)> = match &case.complex_input {
                Some(v) => v.iter().map(|c| (c[0], c[1])).collect(),
                None => return FftObserved::Error("irfftn requires complex_input".to_owned()),
            };
            let shape = match &case.shape {
                Some(s) => s.clone(),
                None => return FftObserved::Error("irfftn requires shape".to_owned()),
            };
            match fsci_fft::irfftn(&input, &shape, &opts) {
                Ok(v) => FftObserved::RealVector(v),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Fftfreq => {
            let n = case.real_input.as_ref().map_or(0, |v| v.len());
            let spacing = case.sample_spacing.unwrap_or(1.0);
            match fsci_fft::fftfreq(n, spacing) {
                Ok(v) => FftObserved::RealVector(v),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Rfftfreq => {
            let n = case.real_input.as_ref().map_or(0, |v| v.len());
            let spacing = case.sample_spacing.unwrap_or(1.0);
            match fsci_fft::rfftfreq(n, spacing) {
                Ok(v) => FftObserved::RealVector(v),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Fftshift => {
            let input = match &case.real_input {
                Some(v) => v.clone(),
                None => return FftObserved::Error("fftshift requires real_input".to_owned()),
            };
            FftObserved::RealVector(fsci_fft::fftshift_1d(&input))
        }
        FftTransformKind::Ifftshift => {
            let input = match &case.real_input {
                Some(v) => v.clone(),
                None => return FftObserved::Error("ifftshift requires real_input".to_owned()),
            };
            FftObserved::RealVector(fsci_fft::ifftshift_1d(&input))
        }
        FftTransformKind::Hfft => {
            let input: Vec<(f64, f64)> = match &case.complex_input {
                Some(v) => v.iter().map(|c| (c[0], c[1])).collect(),
                None => return FftObserved::Error("hfft requires complex_input".to_owned()),
            };
            match fsci_fft::hfft(&input, case.output_len, &opts) {
                Ok(v) => FftObserved::RealVector(v),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Ihfft => {
            let input = match &case.real_input {
                Some(v) => v.as_slice(),
                None => return FftObserved::Error("ihfft requires real_input".to_owned()),
            };
            match fsci_fft::ihfft(input, case.output_len, &opts) {
                Ok(v) => FftObserved::ComplexVector(v.iter().map(|c| [c.0, c.1]).collect()),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Dct
        | FftTransformKind::Idct
        | FftTransformKind::DctI
        | FftTransformKind::DctIii
        | FftTransformKind::DctIv
        | FftTransformKind::DstI
        | FftTransformKind::DstIi
        | FftTransformKind::DstIii
        | FftTransformKind::DstIv => {
            let input = match &case.real_input {
                Some(v) => v.as_slice(),
                None => return FftObserved::Error("dct/dst requires real_input".to_owned()),
            };
            let result = match case.transform {
                FftTransformKind::Dct => fsci_fft::dct(input, &opts),
                FftTransformKind::Idct => fsci_fft::idct(input, &opts),
                FftTransformKind::DctI => fsci_fft::dct_i(input, &opts),
                FftTransformKind::DctIii => fsci_fft::dct_iii(input, &opts),
                FftTransformKind::DctIv => fsci_fft::dct_iv(input, &opts),
                FftTransformKind::DstI => fsci_fft::dst_i(input, &opts),
                FftTransformKind::DstIi => fsci_fft::dst_ii(input, &opts),
                FftTransformKind::DstIii => fsci_fft::dst_iii(input, &opts),
                FftTransformKind::DstIv => fsci_fft::dst_iv(input, &opts),
                _ => return FftObserved::Error("Unexpected transform kind".to_owned()),
            };
            match result {
                Ok(v) => FftObserved::RealVector(v),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Fft2
        | FftTransformKind::Ifft2
        | FftTransformKind::Fftn
        | FftTransformKind::Ifftn => {
            // Multi-dimensional transforms: require complex_input and shape
            let input: Vec<(f64, f64)> = match &case.complex_input {
                Some(v) => v.iter().map(|c| (c[0], c[1])).collect(),
                None => {
                    return FftObserved::Error("nd transforms require complex_input".to_owned());
                }
            };
            let shape = match &case.shape {
                Some(s) => s.clone(),
                None => return FftObserved::Error("nd transforms require shape".to_owned()),
            };
            let result = match case.transform {
                FftTransformKind::Fft2 => {
                    if shape.len() != 2 {
                        return FftObserved::Error("fft2 requires 2D shape".to_owned());
                    }
                    fsci_fft::fft2(&input, (shape[0], shape[1]), &opts)
                }
                FftTransformKind::Ifft2 => {
                    if shape.len() != 2 {
                        return FftObserved::Error("ifft2 requires 2D shape".to_owned());
                    }
                    fsci_fft::ifft2(&input, (shape[0], shape[1]), &opts)
                }
                FftTransformKind::Fftn => fsci_fft::fftn(&input, &shape, &opts),
                FftTransformKind::Ifftn => fsci_fft::ifftn(&input, &shape, &opts),
                _ => return FftObserved::Error("Unsupported FFT transform kind".to_owned()),
            };
            match result {
                Ok(v) => FftObserved::ComplexVector(v.iter().map(|c| [c.0, c.1]).collect()),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
    }
}

fn execute_fft_case_with_differential_audit(
    case: &FftCase,
    audit_ledger: &fsci_fft::SyncSharedAuditLedger,
) -> FftObserved {
    use fsci_fft::{FftOptions, Normalization};

    let mut opts = FftOptions::default().with_mode(case.mode);
    if let Some(norm) = &case.normalization {
        opts = opts.with_normalization(match norm {
            FftNormalization::Forward => Normalization::Forward,
            FftNormalization::Backward => Normalization::Backward,
            FftNormalization::Ortho => Normalization::Ortho,
        });
    }

    match case.transform {
        FftTransformKind::Fft => {
            let input: Vec<(f64, f64)> = match &case.complex_input {
                Some(v) => v.iter().map(|c| (c[0], c[1])).collect(),
                None => return FftObserved::Error("fft requires complex_input".to_owned()),
            };
            match fsci_fft::fft_with_audit(&input, &opts, audit_ledger) {
                Ok(v) => FftObserved::ComplexVector(v.iter().map(|c| [c.0, c.1]).collect()),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Ifft => {
            let input: Vec<(f64, f64)> = match &case.complex_input {
                Some(v) => v.iter().map(|c| (c[0], c[1])).collect(),
                None => return FftObserved::Error("ifft requires complex_input".to_owned()),
            };
            match fsci_fft::ifft_with_audit(&input, &opts, audit_ledger) {
                Ok(v) => FftObserved::ComplexVector(v.iter().map(|c| [c.0, c.1]).collect()),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Rfft => {
            let input = match &case.real_input {
                Some(v) => v.as_slice(),
                None => return FftObserved::Error("rfft requires real_input".to_owned()),
            };
            match fsci_fft::rfft_with_audit(input, &opts, audit_ledger) {
                Ok(v) => FftObserved::ComplexVector(v.iter().map(|c| [c.0, c.1]).collect()),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Irfft => {
            let input: Vec<(f64, f64)> = match &case.complex_input {
                Some(v) => v.iter().map(|c| (c[0], c[1])).collect(),
                None => return FftObserved::Error("irfft requires complex_input".to_owned()),
            };
            match fsci_fft::irfft_with_audit(&input, case.output_len, &opts, audit_ledger) {
                Ok(v) => FftObserved::RealVector(v),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Rfft2 => {
            let input = match &case.real_input {
                Some(v) => v.as_slice(),
                None => return FftObserved::Error("rfft2 requires real_input".to_owned()),
            };
            let shape = match &case.shape {
                Some(s) => s.clone(),
                None => return FftObserved::Error("rfft2 requires shape".to_owned()),
            };
            if shape.len() != 2 {
                return FftObserved::Error("rfft2 requires 2D shape".to_owned());
            }
            match fsci_fft::rfft2_with_audit(input, (shape[0], shape[1]), &opts, audit_ledger) {
                Ok(v) => FftObserved::ComplexVector(v.iter().map(|c| [c.0, c.1]).collect()),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Irfft2 => {
            let input: Vec<(f64, f64)> = match &case.complex_input {
                Some(v) => v.iter().map(|c| (c[0], c[1])).collect(),
                None => return FftObserved::Error("irfft2 requires complex_input".to_owned()),
            };
            let shape = match &case.shape {
                Some(s) => s.clone(),
                None => return FftObserved::Error("irfft2 requires shape".to_owned()),
            };
            if shape.len() != 2 {
                return FftObserved::Error("irfft2 requires 2D shape".to_owned());
            }
            match fsci_fft::irfft2_with_audit(&input, (shape[0], shape[1]), &opts, audit_ledger) {
                Ok(v) => FftObserved::RealVector(v),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Rfftn => {
            let input = match &case.real_input {
                Some(v) => v.as_slice(),
                None => return FftObserved::Error("rfftn requires real_input".to_owned()),
            };
            let shape = match &case.shape {
                Some(s) => s.clone(),
                None => return FftObserved::Error("rfftn requires shape".to_owned()),
            };
            match fsci_fft::rfftn_with_audit(input, &shape, &opts, audit_ledger) {
                Ok(v) => FftObserved::ComplexVector(v.iter().map(|c| [c.0, c.1]).collect()),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Irfftn => {
            let input: Vec<(f64, f64)> = match &case.complex_input {
                Some(v) => v.iter().map(|c| (c[0], c[1])).collect(),
                None => return FftObserved::Error("irfftn requires complex_input".to_owned()),
            };
            let shape = match &case.shape {
                Some(s) => s.clone(),
                None => return FftObserved::Error("irfftn requires shape".to_owned()),
            };
            match fsci_fft::irfftn_with_audit(&input, &shape, &opts, audit_ledger) {
                Ok(v) => FftObserved::RealVector(v),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Hfft => {
            let input: Vec<(f64, f64)> = match &case.complex_input {
                Some(v) => v.iter().map(|c| (c[0], c[1])).collect(),
                None => return FftObserved::Error("hfft requires complex_input".to_owned()),
            };
            match fsci_fft::hfft_with_audit(&input, case.output_len, &opts, audit_ledger) {
                Ok(v) => FftObserved::RealVector(v),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Ihfft => {
            let input = match &case.real_input {
                Some(v) => v.as_slice(),
                None => return FftObserved::Error("ihfft requires real_input".to_owned()),
            };
            match fsci_fft::ihfft_with_audit(input, case.output_len, &opts, audit_ledger) {
                Ok(v) => FftObserved::ComplexVector(v.iter().map(|c| [c.0, c.1]).collect()),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        FftTransformKind::Fft2
        | FftTransformKind::Ifft2
        | FftTransformKind::Fftn
        | FftTransformKind::Ifftn => {
            let input: Vec<(f64, f64)> = match &case.complex_input {
                Some(v) => v.iter().map(|c| (c[0], c[1])).collect(),
                None => {
                    return FftObserved::Error("nd transforms require complex_input".to_owned());
                }
            };
            let shape = match &case.shape {
                Some(s) => s.clone(),
                None => return FftObserved::Error("nd transforms require shape".to_owned()),
            };
            let result = match case.transform {
                FftTransformKind::Fft2 => {
                    if shape.len() != 2 {
                        return FftObserved::Error("fft2 requires 2D shape".to_owned());
                    }
                    fsci_fft::fft2_with_audit(&input, (shape[0], shape[1]), &opts, audit_ledger)
                }
                FftTransformKind::Ifft2 => {
                    if shape.len() != 2 {
                        return FftObserved::Error("ifft2 requires 2D shape".to_owned());
                    }
                    fsci_fft::ifft2_with_audit(&input, (shape[0], shape[1]), &opts, audit_ledger)
                }
                FftTransformKind::Fftn => {
                    fsci_fft::fftn_with_audit(&input, &shape, &opts, audit_ledger)
                }
                FftTransformKind::Ifftn => {
                    fsci_fft::ifftn_with_audit(&input, &shape, &opts, audit_ledger)
                }
                _ => return FftObserved::Error("Unsupported FFT transform kind".to_owned()),
            };
            match result {
                Ok(v) => FftObserved::ComplexVector(v.iter().map(|c| [c.0, c.1]).collect()),
                Err(e) => FftObserved::Error(format!("{e}")),
            }
        }
        _ => execute_fft_case(case),
    }
}

fn compare_fft_outcome(expected: &FftExpectedOutcome, observed: &FftObserved) -> (bool, String) {
    match (expected, observed) {
        (
            FftExpectedOutcome::ComplexVector { values, atol, rtol },
            FftObserved::ComplexVector(got),
        ) => {
            let a = atol.unwrap_or(1.0e-10);
            let r = rtol.unwrap_or(1.0e-8);
            if got.len() != values.len() {
                return (
                    false,
                    format!(
                        "complex vector length mismatch: expected {} got {}",
                        values.len(),
                        got.len()
                    ),
                );
            }
            let max_diff = got
                .iter()
                .zip(values.iter())
                .map(|(g, e)| {
                    let dr = (g[0] - e[0]).abs();
                    let di = (g[1] - e[1]).abs();
                    dr.max(di)
                })
                .fold(0.0_f64, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                });
            let pass = got.iter().zip(values.iter()).all(|(g, e)| {
                allclose_scalar(g[0], e[0], a, r) && allclose_scalar(g[1], e[1], a, r)
            });
            if pass {
                (
                    true,
                    format!("complex vector matched (max_diff={max_diff:.2e})"),
                )
            } else {
                (
                    false,
                    format!(
                        "complex vector mismatch: max_diff={max_diff:.2e}, atol={a:.2e}, rtol={r:.2e}"
                    ),
                )
            }
        }
        (FftExpectedOutcome::RealVector { values, atol, rtol }, FftObserved::RealVector(got)) => {
            let a = atol.unwrap_or(1.0e-10);
            let r = rtol.unwrap_or(1.0e-8);
            if got.len() != values.len() {
                return (
                    false,
                    format!(
                        "real vector length mismatch: expected {} got {}",
                        values.len(),
                        got.len()
                    ),
                );
            }
            let max_diff = got
                .iter()
                .zip(values.iter())
                .map(|(g, e)| (g - e).abs())
                .fold(0.0_f64, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                });
            let pass = got
                .iter()
                .zip(values.iter())
                .all(|(g, e)| allclose_scalar(*g, *e, a, r));
            if pass {
                (
                    true,
                    format!("real vector matched (max_diff={max_diff:.2e})"),
                )
            } else {
                (
                    false,
                    format!(
                        "real vector mismatch: max_diff={max_diff:.2e}, atol={a:.2e}, rtol={r:.2e}"
                    ),
                )
            }
        }
        (FftExpectedOutcome::Error { error }, FftObserved::Error(got)) => {
            let pass = got.contains(error.as_str());
            if pass {
                (true, format!("error matched: {got}"))
            } else {
                (
                    false,
                    format!("error mismatch: expected '{error}', got '{got}'"),
                )
            }
        }
        (FftExpectedOutcome::Error { error }, _) => {
            (false, format!("expected error '{error}' but got success"))
        }
        (_, FftObserved::Error(e)) => (false, format!("unexpected error: {e}")),
        _ => (false, "outcome type mismatch".to_owned()),
    }
}

// ── CASP Runtime packet runner ───────────────────────────────────────

pub fn run_casp_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: CaspPacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute_casp_case(case)
        })) {
            Ok(v) => v,
            Err(payload) => {
                case_results.push(CaseResult {
                    case_id: case.case_id().to_owned(),
                    passed: false,
                    message: format!(
                        "PANIC in execute_casp_case: {}",
                        panic_payload_message(payload)
                    ),
                });
                continue;
            }
        };
        let (passed, message) = compare_casp_outcome(&case.expected, &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

#[derive(Debug)]
enum CaspObserved {
    PolicyAction(String),
    SolverAction(String),
    CalibratorFallback(bool),
    Error(String),
}

fn parse_condition_state(s: &str) -> Option<fsci_runtime::MatrixConditionState> {
    match s {
        "well_conditioned" => Some(fsci_runtime::MatrixConditionState::WellConditioned),
        "moderate_condition" => Some(fsci_runtime::MatrixConditionState::ModerateCondition),
        "ill_conditioned" => Some(fsci_runtime::MatrixConditionState::IllConditioned),
        "near_singular" => Some(fsci_runtime::MatrixConditionState::NearSingular),
        _ => None,
    }
}

fn policy_action_name(action: &fsci_runtime::PolicyAction) -> &'static str {
    match action {
        fsci_runtime::PolicyAction::Allow => "allow",
        fsci_runtime::PolicyAction::FullValidate => "full_validate",
        fsci_runtime::PolicyAction::FailClosed => "fail_closed",
    }
}

fn solver_action_name(action: &fsci_runtime::SolverAction) -> &'static str {
    match action {
        fsci_runtime::SolverAction::DirectLU => "direct_lu",
        fsci_runtime::SolverAction::PivotedQR => "pivoted_qr",
        fsci_runtime::SolverAction::SVDFallback => "svd_fallback",
        fsci_runtime::SolverAction::DiagonalFastPath => "diagonal_fast_path",
        fsci_runtime::SolverAction::TriangularFastPath => "triangular_fast_path",
    }
}

fn execute_casp_case(case: &CaspCase) -> CaspObserved {
    // Per frankenscipy-p71m: read tunables from fixture, defaulting to
    // the legacy magic numbers so existing fixtures keep passing.
    let capacity = case.capacity.unwrap_or(64);
    let alpha = case.alpha.unwrap_or(0.05);
    let calibrator_capacity = case.calibrator_capacity.unwrap_or(200);

    match case.test_kind {
        CaspTestKind::PolicyDecision => {
            let cond = case.condition_signal.unwrap_or(0.0);
            let meta = case.metadata_signal.unwrap_or(0.0);
            let anom = case.anomaly_signal.unwrap_or(0.0);

            let signals = fsci_runtime::DecisionSignals::new(cond, meta, anom);
            let mut controller = fsci_runtime::PolicyController::new(case.mode, capacity);
            let decision = controller.decide(signals);
            CaspObserved::PolicyAction(policy_action_name(&decision.action).to_owned())
        }
        CaspTestKind::SolverSelection => {
            let state_str = match &case.condition_state {
                Some(s) => s.as_str(),
                None => {
                    return CaspObserved::Error(
                        "solver_selection requires condition_state".to_owned(),
                    );
                }
            };
            let state = match parse_condition_state(state_str) {
                Some(s) => s,
                None => {
                    return CaspObserved::Error(format!("unknown condition state: {state_str}"));
                }
            };
            let portfolio = fsci_runtime::SolverPortfolio::new(case.mode, capacity);
            let rcond = match state {
                fsci_runtime::MatrixConditionState::WellConditioned => 1e-2,
                fsci_runtime::MatrixConditionState::ModerateCondition => 1e-6,
                fsci_runtime::MatrixConditionState::IllConditioned => 1e-12,
                fsci_runtime::MatrixConditionState::NearSingular => 1e-18,
            };
            let (action, _posterior, _losses, _loss) = portfolio.select_action(rcond, None);
            CaspObserved::SolverAction(solver_action_name(&action).to_owned())
        }
        CaspTestKind::CalibratorDrift => {
            let observations = match &case.observations {
                Some(obs) => obs.clone(),
                None => {
                    return CaspObserved::Error(
                        "calibrator_drift requires observations".to_owned(),
                    );
                }
            };
            let mut calibrator = fsci_runtime::ConformalCalibrator::new(alpha, calibrator_capacity);
            for obs in &observations {
                calibrator.observe(*obs);
            }
            CaspObserved::CalibratorFallback(calibrator.should_fallback())
        }
    }
}

fn compare_casp_outcome(expected: &CaspExpectedOutcome, observed: &CaspObserved) -> (bool, String) {
    match (expected, observed) {
        (CaspExpectedOutcome::PolicyAction { action }, CaspObserved::PolicyAction(got)) => {
            let pass = action == got;
            if pass {
                (true, format!("policy action matched: {got}"))
            } else {
                (
                    false,
                    format!("policy action mismatch: expected '{action}', got '{got}'"),
                )
            }
        }
        (CaspExpectedOutcome::SolverAction { action }, CaspObserved::SolverAction(got)) => {
            let pass = action == got;
            if pass {
                (true, format!("solver action matched: {got}"))
            } else {
                (
                    false,
                    format!("solver action mismatch: expected '{action}', got '{got}'"),
                )
            }
        }
        (
            CaspExpectedOutcome::CalibratorFallback { should_fallback },
            CaspObserved::CalibratorFallback(got),
        ) => {
            let pass = *should_fallback == *got;
            if pass {
                (true, format!("calibrator fallback matched: {got}"))
            } else {
                (
                    false,
                    format!("calibrator fallback mismatch: expected {should_fallback}, got {got}"),
                )
            }
        }
        (CaspExpectedOutcome::Error { error }, CaspObserved::Error(got)) => {
            let pass = got.contains(error.as_str());
            if pass {
                (true, format!("error matched: {got}"))
            } else {
                (
                    false,
                    format!("error mismatch: expected '{error}', got '{got}'"),
                )
            }
        }
        (CaspExpectedOutcome::Error { error }, _) => {
            (false, format!("expected error '{error}' but got success"))
        }
        (_, CaspObserved::Error(e)) => (false, format!("unexpected error: {e}")),
        _ => (false, "outcome type mismatch".to_owned()),
    }
}

// ══════════════════════════════════════════════════════════════════════
// Cluster Conformance Harness
// ══════════════════════════════════════════════════════════════════════

/// Fixture for cluster conformance testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterPacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<ClusterCase>,
}

/// A single cluster conformance test case.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterCase {
    pub case_id: String,
    pub category: String,
    pub mode: RuntimeMode,
    pub function: String,
    #[serde(default)]
    pub seed: Option<u64>,
    pub args: serde_json::Value,
    pub expected: ClusterExpected,
}

impl ClusterCase {
    fn case_id(&self) -> &str {
        &self.case_id
    }
}

/// Expected outcome for a cluster conformance case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterExpected {
    pub kind: String,
    #[serde(default)]
    pub value: Option<serde_json::Value>,
    #[serde(default)]
    pub codes: Option<Vec<usize>>,
    #[serde(default)]
    pub dists: Option<Vec<f64>>,
    #[serde(default)]
    pub atol: Option<f64>,
    #[serde(default)]
    pub rtol: Option<f64>,
    #[serde(default)]
    pub contract_ref: Option<String>,
    /// Opt in to exact-equality label comparison. Default is
    /// permutation-invariant (first-occurrence canonicalization) because
    /// cluster labelings carry no intrinsic ID — see br-7eaq.
    #[serde(default)]
    pub deterministic_labels: bool,
}

/// Canonicalize cluster labels by first-occurrence order: the first
/// distinct label becomes 0, the next becomes 1, and so on. Two
/// labelings that differ only by cluster-ID permutation produce the
/// same canonical vector.
fn canonicalize_labels_first_occurrence(labels: &[usize]) -> Vec<usize> {
    let mut mapping: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut next_id = 0usize;
    let mut out = Vec::with_capacity(labels.len());
    for &l in labels {
        let id = *mapping.entry(l).or_insert_with(|| {
            let v = next_id;
            next_id += 1;
            v
        });
        out.push(id);
    }
    out
}

fn canonicalize_signed_labels_first_occurrence(labels: &[i64]) -> Vec<i64> {
    let mut mapping: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
    let mut next_id = 0i64;
    let mut out = Vec::with_capacity(labels.len());
    for &label in labels {
        if label < 0 {
            out.push(label);
            continue;
        }
        let id = *mapping.entry(label).or_insert_with(|| {
            let v = next_id;
            next_id += 1;
            v
        });
        out.push(id);
    }
    out
}

#[derive(Debug)]
enum ClusterObserved {
    Scalar(f64),
    Boolean(bool),
    Array1D(Vec<f64>),
    Array2D(Vec<Vec<f64>>),
    Labels(Vec<usize>),
    SignedLabels(Vec<i64>),
    Linkage(Vec<[f64; 4]>),
    VqResult { codes: Vec<usize>, dists: Vec<f64> },
    Error(String),
}

fn execute_cluster_case(case: &ClusterCase) -> ClusterObserved {
    match case.function.as_str() {
        "linkage" => execute_linkage(case),
        "fcluster" => execute_fcluster(case),
        "kmeans" => execute_kmeans(case),
        "dbscan" => execute_dbscan(case),
        "mean_shift" => execute_mean_shift(case),
        "vq" => execute_vq(case),
        "whiten" => execute_whiten(case),
        "cophenet" => execute_cophenet(case),
        "inconsistent" => execute_inconsistent(case),
        "silhouette_score" => execute_silhouette_score(case),
        "adjusted_rand_score" => execute_adjusted_rand_score(case),
        "is_valid_linkage" => execute_is_valid_linkage(case),
        "is_monotonic" => execute_is_monotonic(case),
        "leaves_list" => execute_leaves_list(case),
        "num_obs_linkage" => execute_num_obs_linkage(case),
        _ => ClusterObserved::Error(format!("unknown function: {}", case.function)),
    }
}

fn execute_linkage(case: &ClusterCase) -> ClusterObserved {
    let args = &case.args;
    let data: Vec<Vec<f64>> = match serde_json::from_value(args[0].clone()) {
        Ok(d) => d,
        Err(e) => return ClusterObserved::Error(format!("parse data: {e}")),
    };
    let method_str: String = match serde_json::from_value(args[1].clone()) {
        Ok(m) => m,
        Err(e) => return ClusterObserved::Error(format!("parse method: {e}")),
    };
    let method = match method_str.as_str() {
        "single" => fsci_cluster::LinkageMethod::Single,
        "complete" => fsci_cluster::LinkageMethod::Complete,
        "average" => fsci_cluster::LinkageMethod::Average,
        "ward" => fsci_cluster::LinkageMethod::Ward,
        "weighted" => fsci_cluster::LinkageMethod::Weighted,
        "centroid" => fsci_cluster::LinkageMethod::Centroid,
        "median" => fsci_cluster::LinkageMethod::Median,
        _ => return ClusterObserved::Error(format!("unknown linkage method: {method_str}")),
    };
    match fsci_cluster::linkage(&data, method) {
        Ok(z) => ClusterObserved::Linkage(z),
        Err(e) => ClusterObserved::Error(format!("{e:?}")),
    }
}

fn execute_fcluster(case: &ClusterCase) -> ClusterObserved {
    let args = &case.args;
    let z: Vec<[f64; 4]> = match serde_json::from_value(args[0].clone()) {
        Ok(d) => d,
        Err(e) => return ClusterObserved::Error(format!("parse Z: {e}")),
    };
    let max_clusters: usize = match serde_json::from_value(args[1].clone()) {
        Ok(k) => k,
        Err(e) => return ClusterObserved::Error(format!("parse max_clusters: {e}")),
    };
    match fsci_cluster::fcluster(&z, max_clusters) {
        Ok(labels) => ClusterObserved::Labels(labels),
        Err(e) => ClusterObserved::Error(format!("{e:?}")),
    }
}

fn execute_kmeans(case: &ClusterCase) -> ClusterObserved {
    let args = &case.args;
    let data: Vec<Vec<f64>> = match serde_json::from_value(args[0].clone()) {
        Ok(d) => d,
        Err(e) => return ClusterObserved::Error(format!("parse data: {e}")),
    };
    let k: usize = match serde_json::from_value(args[1].clone()) {
        Ok(k) => k,
        Err(e) => return ClusterObserved::Error(format!("parse k: {e}")),
    };
    let max_iter = match args.get(2) {
        Some(value) => match serde_json::from_value(value.clone()) {
            Ok(max_iter) => max_iter,
            Err(e) => return ClusterObserved::Error(format!("parse max_iter: {e}")),
        },
        None => 100,
    };
    let Some(seed) = case.seed else {
        return ClusterObserved::Error("kmeans requires an explicit seed".to_owned());
    };
    match fsci_cluster::kmeans(&data, k, max_iter, seed) {
        Ok(result) => ClusterObserved::Labels(result.labels),
        Err(e) => ClusterObserved::Error(format!("{e:?}")),
    }
}

fn execute_dbscan(case: &ClusterCase) -> ClusterObserved {
    let args = &case.args;
    let data: Vec<Vec<f64>> = match serde_json::from_value(args[0].clone()) {
        Ok(d) => d,
        Err(e) => return ClusterObserved::Error(format!("parse data: {e}")),
    };
    let eps: f64 = match serde_json::from_value(args[1].clone()) {
        Ok(eps) => eps,
        Err(e) => return ClusterObserved::Error(format!("parse eps: {e}")),
    };
    let min_samples: usize = match serde_json::from_value(args[2].clone()) {
        Ok(min_samples) => min_samples,
        Err(e) => return ClusterObserved::Error(format!("parse min_samples: {e}")),
    };
    match fsci_cluster::dbscan(&data, eps, min_samples) {
        Ok(result) => ClusterObserved::SignedLabels(result.labels),
        Err(e) => ClusterObserved::Error(format!("{e:?}")),
    }
}

fn execute_mean_shift(case: &ClusterCase) -> ClusterObserved {
    let args = &case.args;
    let data: Vec<Vec<f64>> = match serde_json::from_value(args[0].clone()) {
        Ok(d) => d,
        Err(e) => return ClusterObserved::Error(format!("parse data: {e}")),
    };
    let bandwidth: f64 = match serde_json::from_value(args[1].clone()) {
        Ok(bandwidth) => bandwidth,
        Err(e) => return ClusterObserved::Error(format!("parse bandwidth: {e}")),
    };
    let max_iter = match args.get(2) {
        Some(value) => match serde_json::from_value(value.clone()) {
            Ok(max_iter) => max_iter,
            Err(e) => return ClusterObserved::Error(format!("parse max_iter: {e}")),
        },
        None => 100,
    };
    match fsci_cluster::mean_shift(&data, bandwidth, max_iter) {
        Ok((_, labels)) => ClusterObserved::Labels(labels),
        Err(e) => ClusterObserved::Error(format!("{e:?}")),
    }
}

fn execute_vq(case: &ClusterCase) -> ClusterObserved {
    let args = &case.args;
    let data: Vec<Vec<f64>> = match serde_json::from_value(args[0].clone()) {
        Ok(d) => d,
        Err(e) => return ClusterObserved::Error(format!("parse data: {e}")),
    };
    let centroids: Vec<Vec<f64>> = match serde_json::from_value(args[1].clone()) {
        Ok(c) => c,
        Err(e) => return ClusterObserved::Error(format!("parse centroids: {e}")),
    };
    match fsci_cluster::vq(&data, &centroids) {
        Ok((codes, dists)) => ClusterObserved::VqResult { codes, dists },
        Err(e) => ClusterObserved::Error(format!("{e:?}")),
    }
}

fn execute_whiten(case: &ClusterCase) -> ClusterObserved {
    let args = &case.args;
    let data: Vec<Vec<f64>> = match serde_json::from_value(args[0].clone()) {
        Ok(d) => d,
        Err(e) => return ClusterObserved::Error(format!("parse data: {e}")),
    };
    match fsci_cluster::whiten(&data) {
        Ok(w) => ClusterObserved::Array2D(w),
        Err(e) => ClusterObserved::Error(format!("{e:?}")),
    }
}

fn execute_cophenet(case: &ClusterCase) -> ClusterObserved {
    let args = &case.args;
    let z: Vec<[f64; 4]> = match serde_json::from_value(args[0].clone()) {
        Ok(d) => d,
        Err(e) => return ClusterObserved::Error(format!("parse Z: {e}")),
    };
    ClusterObserved::Array1D(fsci_cluster::cophenet(&z))
}

fn execute_inconsistent(case: &ClusterCase) -> ClusterObserved {
    let args = &case.args;
    let z: Vec<[f64; 4]> = match serde_json::from_value(args[0].clone()) {
        Ok(d) => d,
        Err(e) => return ClusterObserved::Error(format!("parse Z: {e}")),
    };
    let depth: usize = match serde_json::from_value(args[1].clone()) {
        Ok(d) => d,
        Err(e) => return ClusterObserved::Error(format!("parse depth: {e}")),
    };
    let result = fsci_cluster::inconsistent(&z, depth);
    ClusterObserved::Array2D(result.iter().map(|r| r.to_vec()).collect())
}

fn execute_silhouette_score(case: &ClusterCase) -> ClusterObserved {
    let args = &case.args;
    let data: Vec<Vec<f64>> = match serde_json::from_value(args[0].clone()) {
        Ok(d) => d,
        Err(e) => return ClusterObserved::Error(format!("parse data: {e}")),
    };
    let labels: Vec<usize> = match serde_json::from_value(args[1].clone()) {
        Ok(l) => l,
        Err(e) => return ClusterObserved::Error(format!("parse labels: {e}")),
    };
    match fsci_cluster::silhouette_score(&data, &labels) {
        Ok(score) => ClusterObserved::Scalar(score),
        Err(e) => ClusterObserved::Error(format!("{e:?}")),
    }
}

fn execute_adjusted_rand_score(case: &ClusterCase) -> ClusterObserved {
    let args = &case.args;
    let labels_true: Vec<usize> = match serde_json::from_value(args[0].clone()) {
        Ok(l) => l,
        Err(e) => return ClusterObserved::Error(format!("parse labels_true: {e}")),
    };
    let labels_pred: Vec<usize> = match serde_json::from_value(args[1].clone()) {
        Ok(l) => l,
        Err(e) => return ClusterObserved::Error(format!("parse labels_pred: {e}")),
    };
    match fsci_cluster::adjusted_rand_score(&labels_true, &labels_pred) {
        Ok(score) => ClusterObserved::Scalar(score),
        Err(e) => ClusterObserved::Error(format!("{e:?}")),
    }
}

fn execute_is_valid_linkage(case: &ClusterCase) -> ClusterObserved {
    let args = &case.args;
    let z: Vec<[f64; 4]> = match serde_json::from_value(args[0].clone()) {
        Ok(d) => d,
        Err(e) => return ClusterObserved::Error(format!("parse Z: {e}")),
    };
    ClusterObserved::Boolean(fsci_cluster::is_valid_linkage(&z))
}

fn execute_is_monotonic(case: &ClusterCase) -> ClusterObserved {
    let args = &case.args;
    let z: Vec<[f64; 4]> = match serde_json::from_value(args[0].clone()) {
        Ok(d) => d,
        Err(e) => return ClusterObserved::Error(format!("parse Z: {e}")),
    };
    ClusterObserved::Boolean(fsci_cluster::is_monotonic(&z))
}

fn execute_leaves_list(case: &ClusterCase) -> ClusterObserved {
    let args = &case.args;
    let z: Vec<[f64; 4]> = match serde_json::from_value(args[0].clone()) {
        Ok(d) => d,
        Err(e) => return ClusterObserved::Error(format!("parse Z: {e}")),
    };
    let leaves = fsci_cluster::leaves_list(&z);
    ClusterObserved::Labels(leaves)
}

fn execute_num_obs_linkage(case: &ClusterCase) -> ClusterObserved {
    let args = &case.args;
    let z: Vec<[f64; 4]> = match serde_json::from_value(args[0].clone()) {
        Ok(d) => d,
        Err(e) => return ClusterObserved::Error(format!("parse Z: {e}")),
    };
    ClusterObserved::Scalar(fsci_cluster::num_obs_linkage(&z) as f64)
}

fn compare_cluster_outcome(case: &ClusterCase, observed: &ClusterObserved) -> (bool, String) {
    let atol = case.expected.atol.unwrap_or(1e-10);
    let rtol = case.expected.rtol.unwrap_or(1e-10);

    match (&case.expected.kind.as_str(), observed) {
        (&"scalar", ClusterObserved::Scalar(got)) => {
            let exp = case
                .expected
                .value
                .as_ref()
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let diff = (exp - got).abs();
            if allclose_scalar(*got, exp, atol, rtol) {
                (true, format!("scalar matched: {got}"))
            } else {
                (
                    false,
                    format!("scalar mismatch: expected {exp}, got {got}, diff {diff}"),
                )
            }
        }
        (&"array", ClusterObserved::Array1D(got)) => {
            let exp: Vec<f64> = case
                .expected
                .value
                .as_ref()
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            if exp.len() != got.len() {
                return (
                    false,
                    format!("array length mismatch: {} vs {}", exp.len(), got.len()),
                );
            }
            for (i, (&e, &g)) in exp.iter().zip(got.iter()).enumerate() {
                if !allclose_scalar(g, e, atol, rtol) {
                    return (
                        false,
                        format!("array mismatch at [{i}]: expected {e}, got {g}"),
                    );
                }
            }
            (true, "array matched".to_string())
        }
        (&"array" | &"matrix", ClusterObserved::Array2D(got)) => {
            let exp: Vec<Vec<f64>> = case
                .expected
                .value
                .as_ref()
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            if exp.len() != got.len() {
                return (
                    false,
                    format!("matrix rows mismatch: {} vs {}", exp.len(), got.len()),
                );
            }
            for (i, (er, gr)) in exp.iter().zip(got.iter()).enumerate() {
                if er.len() != gr.len() {
                    return (false, format!("matrix cols mismatch at row {i}"));
                }
                for (j, (&e, &g)) in er.iter().zip(gr.iter()).enumerate() {
                    if !allclose_scalar(g, e, atol, rtol) {
                        return (
                            false,
                            format!("matrix mismatch at [{i},{j}]: expected {e}, got {g}"),
                        );
                    }
                }
            }
            (true, "matrix matched".to_string())
        }
        (&"labels", ClusterObserved::Labels(got)) => {
            let exp: Vec<usize> = case
                .expected
                .value
                .as_ref()
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            // Cluster labels are permutation-invariant: a k-means or
            // dbscan that names the same partition with different
            // integer IDs is semantically identical. Default to
            // first-occurrence canonicalization (br-7eaq); opt back
            // into exact comparison via `deterministic_labels: true`.
            let (exp_cmp, got_cmp) = if case.expected.deterministic_labels {
                (exp.clone(), got.clone())
            } else {
                (
                    canonicalize_labels_first_occurrence(&exp),
                    canonicalize_labels_first_occurrence(got),
                )
            };
            if exp_cmp == got_cmp {
                (true, format!("labels matched: {got:?}"))
            } else {
                (
                    false,
                    format!("labels mismatch: expected {exp:?}, got {got:?}"),
                )
            }
        }
        (&"signed_labels", ClusterObserved::SignedLabels(got)) => {
            let exp: Vec<i64> = case
                .expected
                .value
                .as_ref()
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            let (exp_cmp, got_cmp) = if case.expected.deterministic_labels {
                (exp.clone(), got.clone())
            } else {
                (
                    canonicalize_signed_labels_first_occurrence(&exp),
                    canonicalize_signed_labels_first_occurrence(got),
                )
            };
            if exp_cmp == got_cmp {
                (true, format!("signed labels matched: {got:?}"))
            } else {
                (
                    false,
                    format!("signed labels mismatch: expected {exp:?}, got {got:?}"),
                )
            }
        }
        (&"linkage", ClusterObserved::Linkage(got)) => {
            let exp: Vec<[f64; 4]> = case
                .expected
                .value
                .as_ref()
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            if exp.len() != got.len() {
                return (
                    false,
                    format!("linkage rows mismatch: {} vs {}", exp.len(), got.len()),
                );
            }
            for (i, (er, gr)) in exp.iter().zip(got.iter()).enumerate() {
                for j in 0..4 {
                    if !allclose_scalar(gr[j], er[j], atol, rtol) {
                        return (
                            false,
                            format!(
                                "linkage mismatch at [{i},{j}]: expected {}, got {}",
                                er[j], gr[j]
                            ),
                        );
                    }
                }
            }
            (true, "linkage matched".to_string())
        }
        (&"vq_result", ClusterObserved::VqResult { codes, dists }) => {
            let exp_codes = case.expected.codes.clone().unwrap_or_default();
            let exp_dists = case.expected.dists.clone().unwrap_or_default();
            if exp_codes != *codes {
                return (
                    false,
                    format!("vq codes mismatch: expected {exp_codes:?}, got {codes:?}"),
                );
            }
            if exp_dists.len() != dists.len() {
                return (false, "vq dists length mismatch".to_string());
            }
            for (i, (&e, &g)) in exp_dists.iter().zip(dists.iter()).enumerate() {
                if !allclose_scalar(g, e, atol, rtol) {
                    return (
                        false,
                        format!("vq dists mismatch at [{i}]: expected {e}, got {g}"),
                    );
                }
            }
            (true, "vq_result matched".to_string())
        }
        (&"boolean", ClusterObserved::Boolean(got)) => {
            let exp = case
                .expected
                .value
                .as_ref()
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if exp == *got {
                (true, format!("boolean matched: {got}"))
            } else {
                (
                    false,
                    format!("boolean mismatch: expected {exp}, got {got}"),
                )
            }
        }
        (_, ClusterObserved::Error(e)) => (false, format!("execution error: {e}")),
        _ => (
            false,
            format!("unexpected observed type for kind '{}'", case.expected.kind),
        ),
    }
}

/// Run the cluster conformance packet.
pub fn run_cluster_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: ClusterPacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute_cluster_case(case)
        })) {
            Ok(v) => v,
            Err(payload) => {
                case_results.push(CaseResult {
                    case_id: case.case_id().to_owned(),
                    passed: false,
                    message: format!(
                        "PANIC in execute_cluster_case: {}",
                        panic_payload_message(payload)
                    ),
                });
                continue;
            }
        };
        let (passed, message) = compare_cluster_outcome(case, &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

// ══════════════════════════════════════════════════════════════════════
// Spatial Conformance Harness
// ══════════════════════════════════════════════════════════════════════

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialPacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<SpatialCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialCase {
    pub case_id: String,
    pub category: String,
    pub mode: RuntimeMode,
    pub function: String,
    pub args: Vec<serde_json::Value>,
    pub expected: SpatialExpected,
}

impl SpatialCase {
    pub fn case_id(&self) -> &str {
        &self.case_id
    }
}

#[derive(Debug, Deserialize)]
pub struct SpatialExpected {
    pub kind: String,
    pub value: Option<serde_json::Value>,
    pub index: Option<usize>,
    pub distance: Option<f64>,
    pub vertices: Option<Vec<usize>>,
    pub area: Option<f64>,
    pub disparity: Option<f64>,
    pub atol: Option<f64>,
    pub rtol: Option<f64>,
    #[serde(default)]
    pub contract_ref: String,
    /// br-nmh2: k-NN parallel arrays. `indices` is the (sorted-by-
    /// distance) list of point indices and `distances` the matching
    /// distances. Used by the `kdtree_query_k_result` and
    /// `kdtree_ball_point_result` expected.kind variants.
    #[serde(default)]
    pub indices: Option<Vec<usize>>,
    #[serde(default)]
    pub distances: Option<Vec<f64>>,
    /// br-uufs: query_ball_tree expected neighbor lists (per-source-
    /// point index list, each sorted ascending).
    #[serde(default)]
    pub neighbors: Option<Vec<Vec<usize>>>,
    /// br-d1jx: permutation-invariant Voronoi summary scalars.
    #[serde(default)]
    pub vertex_count: Option<usize>,
    #[serde(default)]
    pub ridge_count: Option<usize>,
    /// br-d1jx: sorted multiset of Voronoi vertex coordinates,
    /// each as [x, y]. Sort key is lexicographic.
    #[serde(default)]
    pub sorted_vertices: Option<Vec<[f64; 2]>>,
}

#[derive(Debug)]
enum SpatialObserved {
    Scalar(f64),
    Array1D(Vec<f64>),
    Array2D(Vec<Vec<f64>>),
    KdTreeQuery {
        index: usize,
        distance: f64,
    },
    /// k-nearest result: parallel index/distance vectors of length k.
    /// br-nmh2.
    KdTreeQueryK {
        indices: Vec<usize>,
        distances: Vec<f64>,
    },
    /// query_ball_point: indices of points within radius r. br-nmh2.
    KdTreeBallPoint {
        indices: Vec<usize>,
    },
    ConvexHull {
        vertices: Vec<usize>,
        area: f64,
    },
    /// br-uufs: KDTree.query_ball_tree result — list of neighbor
    /// index lists (one per point in tree A; each list contains the
    /// indices in tree B within radius r, sorted ascending).
    KdTreeBallTree {
        neighbors: Vec<Vec<usize>>,
    },
    /// br-d1jx: Voronoi diagram summary. Region/ridge ordering is
    /// permutation-sensitive between scipy and fsci, so the comparator
    /// uses permutation-invariant scalars (vertex_count, ridge_count)
    /// plus the sorted multiset of vertex coordinates.
    Voronoi {
        vertex_count: usize,
        ridge_count: usize,
        sorted_vertices: Vec<[f64; 2]>,
    },
    HalfspaceIntersection {
        intersections: Vec<Vec<f64>>,
        dual_points: Vec<Vec<f64>>,
        dual_vertices: Vec<usize>,
        dual_area: f64,
        dual_volume: f64,
        is_bounded: bool,
    },
    Procrustes {
        disparity: f64,
    },
    Error(String),
}

fn execute_spatial_case(case: &SpatialCase) -> SpatialObserved {
    match case.function.as_str() {
        "euclidean" | "cityblock" | "chebyshev" | "cosine" | "correlation" | "hamming"
        | "jaccard" | "canberra" | "braycurtis" | "sqeuclidean" => execute_distance_metric(case),
        "minkowski" => execute_minkowski(case),
        "seuclidean" => execute_seuclidean(case),
        "mahalanobis" => execute_mahalanobis(case),
        "wminkowski" => execute_wminkowski(case),
        "pdist" => execute_pdist(case),
        "cdist" => execute_cdist(case),
        "squareform_to_matrix" => execute_squareform_to_matrix(case),
        "kdtree_query" => execute_kdtree_query(case),
        "kdtree_query_k" => execute_kdtree_query_k(case),
        "kdtree_query_ball_point" => execute_kdtree_query_ball_point(case),
        "kdtree_query_pairs" => execute_kdtree_query_pairs(case),
        "kdtree_query_ball_tree" => execute_kdtree_query_ball_tree(case),
        "kdtree_count_neighbors" => execute_kdtree_count_neighbors(case),
        "directed_hausdorff" => execute_directed_hausdorff(case),
        "convex_hull" => execute_convex_hull(case),
        "voronoi" => execute_voronoi(case),
        "halfspace_intersection" => execute_halfspace_intersection(case),
        "procrustes" => execute_procrustes(case),
        "geometric_slerp" => execute_geometric_slerp(case),
        _ => SpatialObserved::Error(format!("unknown function: {}", case.function)),
    }
}

fn execute_distance_metric(case: &SpatialCase) -> SpatialObserved {
    let a: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse a: {e}")),
    };
    let b: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse b: {e}")),
    };
    let result = match case.function.as_str() {
        "euclidean" => fsci_spatial::euclidean(&a, &b),
        "cityblock" => fsci_spatial::cityblock(&a, &b),
        "chebyshev" => fsci_spatial::chebyshev(&a, &b),
        "cosine" => fsci_spatial::cosine(&a, &b),
        "correlation" => fsci_spatial::correlation(&a, &b),
        "hamming" => fsci_spatial::hamming(&a, &b),
        "jaccard" => fsci_spatial::jaccard(&a, &b),
        "canberra" => fsci_spatial::canberra(&a, &b),
        "braycurtis" => fsci_spatial::braycurtis(&a, &b),
        "sqeuclidean" => fsci_spatial::sqeuclidean(&a, &b),
        _ => return SpatialObserved::Error(format!("unknown metric: {}", case.function)),
    };
    SpatialObserved::Scalar(result)
}

/// br-nmh2: scipy.spatial.distance.minkowski(a, b, p).
fn execute_minkowski(case: &SpatialCase) -> SpatialObserved {
    let a: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse a: {e}")),
    };
    let b: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse b: {e}")),
    };
    let p: f64 = match serde_json::from_value(case.args[2].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse p: {e}")),
    };
    SpatialObserved::Scalar(fsci_spatial::minkowski(&a, &b, p))
}

/// br-nmh2: scipy.spatial.distance.seuclidean(u, v, V).
fn execute_seuclidean(case: &SpatialCase) -> SpatialObserved {
    let a: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse a: {e}")),
    };
    let b: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse b: {e}")),
    };
    let v: Vec<f64> = match serde_json::from_value(case.args[2].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse v: {e}")),
    };
    SpatialObserved::Scalar(fsci_spatial::seuclidean(&a, &b, &v))
}

/// br-nmh2: scipy.spatial.distance.mahalanobis(u, v, VI).
fn execute_mahalanobis(case: &SpatialCase) -> SpatialObserved {
    let a: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse a: {e}")),
    };
    let b: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse b: {e}")),
    };
    let vi: Vec<Vec<f64>> = match serde_json::from_value(case.args[2].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse vi: {e}")),
    };
    SpatialObserved::Scalar(fsci_spatial::mahalanobis(&a, &b, &vi))
}

/// br-nmh2: scipy.spatial.distance.wminkowski(u, v, p, w).
fn execute_wminkowski(case: &SpatialCase) -> SpatialObserved {
    let a: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse a: {e}")),
    };
    let b: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse b: {e}")),
    };
    let p: f64 = match serde_json::from_value(case.args[2].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse p: {e}")),
    };
    let w: Vec<f64> = match serde_json::from_value(case.args[3].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse w: {e}")),
    };
    SpatialObserved::Scalar(fsci_spatial::wminkowski(&a, &b, p, &w))
}

fn execute_pdist(case: &SpatialCase) -> SpatialObserved {
    let data: Vec<Vec<f64>> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse data: {e}")),
    };
    let metric_str: String = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse metric: {e}")),
    };
    let metric = match metric_str.as_str() {
        "euclidean" => fsci_spatial::DistanceMetric::Euclidean,
        "sqeuclidean" => fsci_spatial::DistanceMetric::SqEuclidean,
        "cityblock" => fsci_spatial::DistanceMetric::Cityblock,
        "chebyshev" => fsci_spatial::DistanceMetric::Chebyshev,
        "cosine" => fsci_spatial::DistanceMetric::Cosine,
        "correlation" => fsci_spatial::DistanceMetric::Correlation,
        "hamming" => fsci_spatial::DistanceMetric::Hamming,
        "jaccard" => fsci_spatial::DistanceMetric::Jaccard,
        "canberra" => fsci_spatial::DistanceMetric::Canberra,
        "braycurtis" => fsci_spatial::DistanceMetric::Braycurtis,
        _ => return SpatialObserved::Error(format!("unknown metric: {metric_str}")),
    };
    match fsci_spatial::pdist(&data, metric) {
        Ok(d) => SpatialObserved::Array1D(d),
        Err(e) => SpatialObserved::Error(format!("{e:?}")),
    }
}

fn execute_cdist(case: &SpatialCase) -> SpatialObserved {
    let xa: Vec<Vec<f64>> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse xa: {e}")),
    };
    let xb: Vec<Vec<f64>> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse xb: {e}")),
    };
    let metric_str: String = match serde_json::from_value(case.args[2].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse metric: {e}")),
    };
    let metric = match metric_str.as_str() {
        "euclidean" => fsci_spatial::DistanceMetric::Euclidean,
        "sqeuclidean" => fsci_spatial::DistanceMetric::SqEuclidean,
        "cityblock" => fsci_spatial::DistanceMetric::Cityblock,
        "chebyshev" => fsci_spatial::DistanceMetric::Chebyshev,
        "cosine" => fsci_spatial::DistanceMetric::Cosine,
        "correlation" => fsci_spatial::DistanceMetric::Correlation,
        "hamming" => fsci_spatial::DistanceMetric::Hamming,
        "jaccard" => fsci_spatial::DistanceMetric::Jaccard,
        "canberra" => fsci_spatial::DistanceMetric::Canberra,
        "braycurtis" => fsci_spatial::DistanceMetric::Braycurtis,
        _ => return SpatialObserved::Error(format!("unknown metric: {metric_str}")),
    };
    match fsci_spatial::cdist_metric(&xa, &xb, metric) {
        Ok(d) => SpatialObserved::Array2D(d),
        Err(e) => SpatialObserved::Error(format!("{e:?}")),
    }
}

fn execute_squareform_to_matrix(case: &SpatialCase) -> SpatialObserved {
    let condensed: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse condensed: {e}")),
    };
    match fsci_spatial::squareform_to_matrix(&condensed) {
        Ok(m) => SpatialObserved::Array2D(m),
        Err(e) => SpatialObserved::Error(format!("{e:?}")),
    }
}

fn execute_kdtree_query(case: &SpatialCase) -> SpatialObserved {
    let data: Vec<Vec<f64>> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse data: {e}")),
    };
    let query: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse query: {e}")),
    };
    let tree = match fsci_spatial::KDTree::new(&data) {
        Ok(t) => t,
        Err(e) => return SpatialObserved::Error(format!("build tree: {e:?}")),
    };
    match tree.query(&query) {
        Ok((index, distance)) => SpatialObserved::KdTreeQuery { index, distance },
        Err(e) => SpatialObserved::Error(format!("{e:?}")),
    }
}

/// br-nmh2: k-nearest-neighbor query (k > 1).
fn execute_kdtree_query_k(case: &SpatialCase) -> SpatialObserved {
    let data: Vec<Vec<f64>> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse data: {e}")),
    };
    let query: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse query: {e}")),
    };
    let k: usize = match serde_json::from_value(case.args[2].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse k: {e}")),
    };
    let tree = match fsci_spatial::KDTree::new(&data) {
        Ok(t) => t,
        Err(e) => return SpatialObserved::Error(format!("build tree: {e:?}")),
    };
    match tree.query_k(&query, k) {
        Ok(pairs) => {
            let indices = pairs.iter().map(|p| p.0).collect();
            let distances = pairs.iter().map(|p| p.1).collect();
            SpatialObserved::KdTreeQueryK { indices, distances }
        }
        Err(e) => SpatialObserved::Error(format!("{e:?}")),
    }
}

/// br-nmh2: query_ball_point — all indices within radius r.
fn execute_kdtree_query_ball_point(case: &SpatialCase) -> SpatialObserved {
    let data: Vec<Vec<f64>> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse data: {e}")),
    };
    let query: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse query: {e}")),
    };
    let r: f64 = match serde_json::from_value(case.args[2].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse r: {e}")),
    };
    let tree = match fsci_spatial::KDTree::new(&data) {
        Ok(t) => t,
        Err(e) => return SpatialObserved::Error(format!("build tree: {e:?}")),
    };
    match tree.query_ball_point(&query, r) {
        Ok(mut indices) => {
            indices.sort_unstable();
            SpatialObserved::KdTreeBallPoint { indices }
        }
        Err(e) => SpatialObserved::Error(format!("{e:?}")),
    }
}

// br-uufs: KDTree.query_ball_tree — cross-tree intersection.
// Schema: args = [tree_a_points, tree_b_points, r].
fn execute_kdtree_query_ball_tree(case: &SpatialCase) -> SpatialObserved {
    let data_a: Vec<Vec<f64>> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse data_a: {e}")),
    };
    let data_b: Vec<Vec<f64>> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse data_b: {e}")),
    };
    let r: f64 = match serde_json::from_value(case.args[2].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse r: {e}")),
    };
    let tree_a = match fsci_spatial::KDTree::new(&data_a) {
        Ok(t) => t,
        Err(e) => return SpatialObserved::Error(format!("build tree_a: {e:?}")),
    };
    let tree_b = match fsci_spatial::KDTree::new(&data_b) {
        Ok(t) => t,
        Err(e) => return SpatialObserved::Error(format!("build tree_b: {e:?}")),
    };
    match tree_a.query_ball_tree(&tree_b, r) {
        Ok(nbrs) => {
            let mut sorted: Vec<Vec<usize>> = nbrs
                .into_iter()
                .map(|mut v| {
                    v.sort_unstable();
                    v
                })
                .collect();
            // Outer-list order is fixed (tree_a iteration order); inner
            // is sort_unstable for canonical comparison.
            for v in sorted.iter_mut() {
                v.sort_unstable();
            }
            SpatialObserved::KdTreeBallTree { neighbors: sorted }
        }
        Err(e) => SpatialObserved::Error(format!("{e:?}")),
    }
}

// br-q3ut: KDTree.query_pairs — within-tree close pairs.
// Schema: args = [points: Vec<Vec<f64>>, r: f64].
// Returns Array1D of flat (i, j, i, j, ...) indices — caller sorts before
// equality compare since pair order is not canonical between scipy and fsci.
fn execute_kdtree_query_pairs(case: &SpatialCase) -> SpatialObserved {
    let data: Vec<Vec<f64>> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse data: {e}")),
    };
    let r: f64 = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse r: {e}")),
    };
    let tree = match fsci_spatial::KDTree::new(&data) {
        Ok(t) => t,
        Err(e) => return SpatialObserved::Error(format!("build tree: {e:?}")),
    };
    match tree.query_pairs(r) {
        Ok(pairs) => {
            // Canonicalize each pair as (min, max), sort the multiset.
            let mut canon: Vec<(usize, usize)> = pairs
                .into_iter()
                .map(|(a, b)| if a <= b { (a, b) } else { (b, a) })
                .collect();
            canon.sort_unstable();
            let mut indices = Vec::with_capacity(canon.len());
            for (a, b) in canon {
                indices.push(a);
                indices.push(b);
            }
            SpatialObserved::KdTreeBallPoint { indices }
        }
        Err(e) => SpatialObserved::Error(format!("{e:?}")),
    }
}

// br-q3ut: KDTree.count_neighbors — cross-tree count within radius.
// Schema: args = [tree_a_points, tree_b_points, r].
fn execute_kdtree_count_neighbors(case: &SpatialCase) -> SpatialObserved {
    let data_a: Vec<Vec<f64>> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse data_a: {e}")),
    };
    let data_b: Vec<Vec<f64>> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse data_b: {e}")),
    };
    let r: f64 = match serde_json::from_value(case.args[2].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse r: {e}")),
    };
    let tree_a = match fsci_spatial::KDTree::new(&data_a) {
        Ok(t) => t,
        Err(e) => return SpatialObserved::Error(format!("build tree_a: {e:?}")),
    };
    let tree_b = match fsci_spatial::KDTree::new(&data_b) {
        Ok(t) => t,
        Err(e) => return SpatialObserved::Error(format!("build tree_b: {e:?}")),
    };
    match tree_a.count_neighbors(&tree_b, r) {
        Ok(c) => SpatialObserved::Scalar(c as f64),
        Err(e) => SpatialObserved::Error(format!("{e:?}")),
    }
}

fn execute_directed_hausdorff(case: &SpatialCase) -> SpatialObserved {
    let xa: Vec<Vec<f64>> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse xa: {e}")),
    };
    let xb: Vec<Vec<f64>> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse xb: {e}")),
    };
    match fsci_spatial::directed_hausdorff(&xa, &xb) {
        Ok(d) => SpatialObserved::Scalar(d),
        Err(e) => SpatialObserved::Error(format!("{e:?}")),
    }
}

fn execute_convex_hull(case: &SpatialCase) -> SpatialObserved {
    let points: Vec<Vec<f64>> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse points: {e}")),
    };
    let points_2d: Vec<(f64, f64)> = points.iter().map(|p| (p[0], p[1])).collect();
    match fsci_spatial::ConvexHull::new(&points_2d) {
        Ok(hull) => {
            let mut vertices = hull.vertices.clone();
            vertices.sort();
            SpatialObserved::ConvexHull {
                vertices,
                area: hull.area,
            }
        }
        Err(e) => SpatialObserved::Error(format!("{e:?}")),
    }
}

/// br-d1jx: 2D Voronoi diagram. Reduces the (vertices, ridges,
/// regions, point_region) tuple to permutation-invariant counts plus
/// a lexicographically-sorted vertex coordinate multiset, since the
/// per-point region indices and per-ridge orderings carry no
/// canonical sort.
fn execute_voronoi(case: &SpatialCase) -> SpatialObserved {
    let points: Vec<Vec<f64>> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse points: {e}")),
    };
    let points_2d: Vec<(f64, f64)> = points.iter().map(|p| (p[0], p[1])).collect();
    match fsci_spatial::Voronoi::new(&points_2d) {
        Ok(v) => {
            let mut sorted_vertices: Vec<[f64; 2]> =
                v.vertices.iter().map(|(x, y)| [*x, *y]).collect();
            sorted_vertices.sort_by(|a, b| {
                a[0].partial_cmp(&b[0])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a[1].partial_cmp(&b[1]).unwrap_or(std::cmp::Ordering::Equal))
            });
            SpatialObserved::Voronoi {
                vertex_count: v.vertices.len(),
                ridge_count: v.ridge_points.len(),
                sorted_vertices,
            }
        }
        Err(e) => SpatialObserved::Error(format!("{e:?}")),
    }
}

fn execute_halfspace_intersection(case: &SpatialCase) -> SpatialObserved {
    let halfspaces: Vec<Vec<f64>> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse halfspaces: {e}")),
    };
    let interior_point: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse interior_point: {e}")),
    };

    match fsci_spatial::HalfspaceIntersection::from_nd(&halfspaces, &interior_point) {
        Ok(result) => SpatialObserved::HalfspaceIntersection {
            intersections: result.intersections,
            dual_points: result.dual_points,
            dual_vertices: result.dual_vertices,
            dual_area: result.dual_area,
            dual_volume: result.dual_volume,
            is_bounded: result.is_bounded,
        },
        Err(e) => SpatialObserved::Error(format!("{e:?}")),
    }
}

fn execute_procrustes(case: &SpatialCase) -> SpatialObserved {
    let data1: Vec<Vec<f64>> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse data1: {e}")),
    };
    let data2: Vec<Vec<f64>> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse data2: {e}")),
    };
    match fsci_spatial::procrustes(&data1, &data2) {
        Ok(result) => SpatialObserved::Procrustes {
            disparity: result.disparity,
        },
        Err(e) => SpatialObserved::Error(format!("{e:?}")),
    }
}

fn execute_geometric_slerp(case: &SpatialCase) -> SpatialObserved {
    let start: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse start: {e}")),
    };
    let end: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse end: {e}")),
    };
    let t_values: Vec<f64> = match serde_json::from_value(case.args[2].clone()) {
        Ok(v) => v,
        Err(e) => return SpatialObserved::Error(format!("parse t_values: {e}")),
    };
    match fsci_spatial::geometric_slerp(&start, &end, &t_values) {
        Ok(result) => SpatialObserved::Array2D(result),
        Err(e) => SpatialObserved::Error(format!("{e:?}")),
    }
}

fn compare_spatial_outcome(case: &SpatialCase, observed: &SpatialObserved) -> (bool, String) {
    let atol = case.expected.atol.unwrap_or(1e-10);
    let rtol = case.expected.rtol.unwrap_or(1e-10);

    match (case.expected.kind.as_str(), observed) {
        ("scalar", SpatialObserved::Scalar(got)) => {
            let expected = case
                .expected
                .value
                .as_ref()
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let diff = (got - expected).abs();
            if allclose_scalar(*got, expected, atol, rtol) {
                (true, format!("scalar match: {got}"))
            } else {
                (
                    false,
                    format!("scalar mismatch: got {got}, expected {expected}, diff {diff}"),
                )
            }
        }
        ("array", SpatialObserved::Array1D(got)) => {
            let expected: Vec<f64> = case
                .expected
                .value
                .as_ref()
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            if got.len() != expected.len() {
                return (
                    false,
                    format!(
                        "length mismatch: got {}, expected {}",
                        got.len(),
                        expected.len()
                    ),
                );
            }
            for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
                if !allclose_scalar(g, e, atol, rtol) {
                    return (false, format!("array[{i}] mismatch: got {g}, expected {e}"));
                }
            }
            (true, format!("array match ({} elements)", got.len()))
        }
        ("matrix", SpatialObserved::Array2D(got)) => {
            let expected: Vec<Vec<f64>> = case
                .expected
                .value
                .as_ref()
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            if got.len() != expected.len() {
                return (
                    false,
                    format!(
                        "row count mismatch: got {}, expected {}",
                        got.len(),
                        expected.len()
                    ),
                );
            }
            for (i, (grow, erow)) in got.iter().zip(expected.iter()).enumerate() {
                if grow.len() != erow.len() {
                    return (false, format!("row[{i}] length mismatch"));
                }
                for (j, (&g, &e)) in grow.iter().zip(erow.iter()).enumerate() {
                    if !allclose_scalar(g, e, atol, rtol) {
                        return (
                            false,
                            format!("matrix[{i}][{j}] mismatch: got {g}, expected {e}"),
                        );
                    }
                }
            }
            (
                true,
                format!(
                    "matrix match ({}x{})",
                    got.len(),
                    got.first().map_or(0, |r| r.len())
                ),
            )
        }
        ("kdtree_query_result", SpatialObserved::KdTreeQuery { index, distance }) => {
            let exp_idx = case.expected.index.unwrap_or(usize::MAX);
            let exp_dist = case.expected.distance.unwrap_or(0.0);
            if *index != exp_idx {
                return (
                    false,
                    format!("index mismatch: got {index}, expected {exp_idx}"),
                );
            }
            if !allclose_scalar(*distance, exp_dist, atol, rtol) {
                return (
                    false,
                    format!("distance mismatch: got {distance}, expected {exp_dist}"),
                );
            }
            (
                true,
                format!("kdtree query match: idx={index}, dist={distance}"),
            )
        }
        ("kdtree_query_k_result", SpatialObserved::KdTreeQueryK { indices, distances }) => {
            // br-nmh2: parallel arrays. Indices must match exactly
            // (sorted by distance); distances must match within tol.
            let exp_idx = case.expected.indices.as_ref().cloned().unwrap_or_default();
            let exp_dist = case
                .expected
                .distances
                .as_ref()
                .cloned()
                .unwrap_or_default();
            if indices.len() != exp_idx.len() {
                return (
                    false,
                    format!(
                        "k length mismatch: got {} indices, expected {}",
                        indices.len(),
                        exp_idx.len()
                    ),
                );
            }
            for (i, (g, e)) in indices.iter().zip(exp_idx.iter()).enumerate() {
                if g != e {
                    return (
                        false,
                        format!("kdtree_query_k indices[{i}] mismatch: got {g}, expected {e}"),
                    );
                }
            }
            for (i, (g, e)) in distances.iter().zip(exp_dist.iter()).enumerate() {
                if !allclose_scalar(*g, *e, atol, rtol) {
                    return (
                        false,
                        format!("kdtree_query_k distances[{i}] mismatch: got {g}, expected {e}"),
                    );
                }
            }
            (true, format!("kdtree query_k match: k={}", indices.len()))
        }
        ("kdtree_ball_point_result", SpatialObserved::KdTreeBallPoint { indices }) => {
            let mut exp_idx = case.expected.indices.as_ref().cloned().unwrap_or_default();
            exp_idx.sort_unstable();
            if indices != &exp_idx {
                return (
                    false,
                    format!(
                        "kdtree ball_point indices mismatch: got {indices:?}, expected {exp_idx:?}"
                    ),
                );
            }
            (
                true,
                format!("kdtree ball_point match: {} hits", indices.len()),
            )
        }
        ("convex_hull", SpatialObserved::ConvexHull { vertices, area }) => {
            let exp_vertices = case.expected.vertices.as_ref().cloned().unwrap_or_default();
            let exp_area = case.expected.area.unwrap_or(0.0);
            if *vertices != exp_vertices {
                return (
                    false,
                    format!("vertices mismatch: got {vertices:?}, expected {exp_vertices:?}"),
                );
            }
            if !allclose_scalar(*area, exp_area, atol, rtol) {
                return (
                    false,
                    format!("area mismatch: got {area}, expected {exp_area}"),
                );
            }
            (
                true,
                format!("convex hull match: vertices={vertices:?}, area={area}"),
            )
        }
        ("kdtree_ball_tree_result", SpatialObserved::KdTreeBallTree { neighbors }) => {
            let exp = case
                .expected
                .neighbors
                .as_ref()
                .cloned()
                .unwrap_or_default();
            if exp.len() != neighbors.len() {
                return (
                    false,
                    format!(
                        "ball_tree neighbors len mismatch: got {} expected {}",
                        neighbors.len(),
                        exp.len()
                    ),
                );
            }
            for (i, (got, expected)) in neighbors.iter().zip(exp.iter()).enumerate() {
                if got != expected {
                    return (
                        false,
                        format!("ball_tree row {i} mismatch: got {got:?} expected {expected:?}"),
                    );
                }
            }
            (
                true,
                format!("kdtree ball_tree match ({} rows)", neighbors.len()),
            )
        }
        (
            "voronoi_result",
            SpatialObserved::Voronoi {
                vertex_count,
                ridge_count,
                sorted_vertices,
            },
        ) => {
            let exp_vc = case.expected.vertex_count.unwrap_or(0);
            let exp_rc = case.expected.ridge_count.unwrap_or(0);
            if *vertex_count != exp_vc {
                return (
                    false,
                    format!("vertex_count mismatch: got {vertex_count}, expected {exp_vc}"),
                );
            }
            if *ridge_count != exp_rc {
                return (
                    false,
                    format!("ridge_count mismatch: got {ridge_count}, expected {exp_rc}"),
                );
            }
            if let Some(exp_vs) = case.expected.sorted_vertices.as_ref() {
                if exp_vs.len() != sorted_vertices.len() {
                    return (
                        false,
                        format!(
                            "sorted_vertices len mismatch: got {} expected {}",
                            sorted_vertices.len(),
                            exp_vs.len()
                        ),
                    );
                }
                for (got, exp) in sorted_vertices.iter().zip(exp_vs.iter()) {
                    if !allclose_scalar(got[0], exp[0], atol, rtol)
                        || !allclose_scalar(got[1], exp[1], atol, rtol)
                    {
                        return (
                            false,
                            format!("sorted_vertices coord mismatch: got {got:?} expected {exp:?}"),
                        );
                    }
                }
            }
            (
                true,
                format!("voronoi match: vc={vertex_count}, rc={ridge_count}"),
            )
        }
        ("halfspace_intersection", obs @ SpatialObserved::HalfspaceIntersection { .. }) => {
            compare_halfspace_intersection_outcome(case, obs, atol, rtol)
        }
        ("procrustes_result", SpatialObserved::Procrustes { disparity }) => {
            let exp_disparity = case.expected.disparity.unwrap_or(0.0);
            if !allclose_scalar(*disparity, exp_disparity, atol, rtol) {
                return (
                    false,
                    format!("disparity mismatch: got {disparity}, expected {exp_disparity}"),
                );
            }
            (true, format!("procrustes match: disparity={disparity}"))
        }
        ("error_contains", SpatialObserved::Error(e)) => {
            let expected = case
                .expected
                .value
                .as_ref()
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            if e.contains(expected) {
                (true, format!("expected error observed: {expected}"))
            } else {
                (
                    false,
                    format!("error mismatch: got {e:?}, expected substring {expected:?}"),
                )
            }
        }
        (_, SpatialObserved::Error(e)) => (false, format!("execution error: {e}")),
        (kind, obs) => (
            false,
            format!("type mismatch: expected {kind}, got {obs:?}"),
        ),
    }
}

fn compare_halfspace_intersection_outcome(
    case: &SpatialCase,
    observed: &SpatialObserved,
    atol: f64,
    rtol: f64,
) -> (bool, String) {
    let SpatialObserved::HalfspaceIntersection {
        intersections,
        dual_points,
        dual_vertices,
        dual_area,
        dual_volume,
        is_bounded,
    } = observed
    else {
        return (false, format!("type mismatch: got {observed:?}"));
    };

    let Some(expected) = case
        .expected
        .value
        .as_ref()
        .and_then(|value| value.as_object())
    else {
        return (
            false,
            "halfspace_intersection expected value must be an object".to_string(),
        );
    };

    if let Some(expected_bounded) = expected.get("is_bounded").and_then(|value| value.as_bool())
        && *is_bounded != expected_bounded
    {
        return (
            false,
            format!(
                "is_bounded mismatch: got {}, expected {expected_bounded}",
                *is_bounded
            ),
        );
    }

    for (field, got) in [("dual_area", *dual_area), ("dual_volume", *dual_volume)] {
        if let Some(expected_value) = expected.get(field).and_then(|value| value.as_f64()) {
            let diff = (got - expected_value).abs();
            if !allclose_scalar(got, expected_value, atol, rtol) {
                return (
                    false,
                    format!("{field} mismatch: got {got}, expected {expected_value}, diff {diff}"),
                );
            }
        }
    }

    if let Some(expected_vertices) = expected
        .get("dual_vertices")
        .and_then(|value| serde_json::from_value::<Vec<usize>>(value.clone()).ok())
        && dual_vertices != expected_vertices.as_slice()
    {
        return (
            false,
            format!(
                "dual_vertices mismatch: got {dual_vertices:?}, expected {expected_vertices:?}"
            ),
        );
    }

    if let Some(expected_dual_points) = expected
        .get("dual_points")
        .and_then(|value| serde_json::from_value::<Vec<Vec<f64>>>(value.clone()).ok())
        && let Err(message) = compare_ordered_matrix(
            dual_points,
            &expected_dual_points,
            atol,
            rtol,
            "dual_points",
        )
    {
        return (false, message);
    }

    if let Some(expected_intersections) = expected
        .get("intersections")
        .and_then(|value| serde_json::from_value::<Vec<Vec<f64>>>(value.clone()).ok())
        && let Err(message) = compare_unordered_points(
            intersections,
            &expected_intersections,
            atol,
            rtol,
            "intersections",
        )
    {
        return (false, message);
    }

    if let Some(expected_finite_intersections) = expected
        .get("finite_intersections")
        .and_then(|value| serde_json::from_value::<Vec<Vec<f64>>>(value.clone()).ok())
    {
        let finite = intersections
            .iter()
            .filter(|row| row.iter().all(|value| value.is_finite()))
            .cloned()
            .collect::<Vec<_>>();
        if let Err(message) = compare_unordered_points(
            &finite,
            &expected_finite_intersections,
            atol,
            rtol,
            "finite_intersections",
        ) {
            return (false, message);
        }
    }

    if let Some(expected_nonfinite) = expected
        .get("has_nonfinite_intersection")
        .and_then(|value| value.as_bool())
    {
        let has_nonfinite = intersections
            .iter()
            .any(|row| row.iter().any(|value| !value.is_finite()));
        if has_nonfinite != expected_nonfinite {
            return (
                false,
                format!(
                    "has_nonfinite_intersection mismatch: got {has_nonfinite}, expected {expected_nonfinite}"
                ),
            );
        }
    }

    (
        true,
        format!(
            "halfspace intersection match: {} intersections, {} dual points",
            intersections.len(),
            dual_points.len()
        ),
    )
}

fn compare_ordered_matrix(
    got: &[Vec<f64>],
    expected: &[Vec<f64>],
    atol: f64,
    rtol: f64,
    label: &str,
) -> Result<(), String> {
    if got.len() != expected.len() {
        return Err(format!(
            "{label} row count mismatch: got {}, expected {}",
            got.len(),
            expected.len()
        ));
    }
    for (row_idx, (got_row, expected_row)) in got.iter().zip(expected.iter()).enumerate() {
        if got_row.len() != expected_row.len() {
            return Err(format!("{label}[{row_idx}] length mismatch"));
        }
        for (col_idx, (&g, &e)) in got_row.iter().zip(expected_row.iter()).enumerate() {
            if !allclose_scalar(g, e, atol, rtol) {
                return Err(format!(
                    "{label}[{row_idx}][{col_idx}] mismatch: got {g}, expected {e}"
                ));
            }
        }
    }
    Ok(())
}

fn compare_unordered_points(
    got: &[Vec<f64>],
    expected: &[Vec<f64>],
    atol: f64,
    rtol: f64,
    label: &str,
) -> Result<(), String> {
    if got.len() != expected.len() {
        return Err(format!(
            "{label} length mismatch: got {}, expected {}",
            got.len(),
            expected.len()
        ));
    }

    // Maximum bipartite matching via DFS augmenting paths (Hungarian
    // matching phase). Greedy `position(...)` fails when an early
    // expected point consumes a `got` candidate that is the ONLY
    // within-tolerance match for a later expected point — see br-cwgm.
    // This runs in O(E * V); conformance point sets are small (<100).
    let n = expected.len();
    // adj[e] = list of got indices within tolerance of expected[e].
    let mut adj: Vec<Vec<usize>> = Vec::with_capacity(n);
    for ep in expected {
        let mut row = Vec::new();
        for (idx, gp) in got.iter().enumerate() {
            if points_close(gp, ep, atol, rtol) {
                row.push(idx);
            }
        }
        adj.push(row);
    }

    // match_of_got[g] = expected index currently matched to got[g], or None.
    let mut match_of_got: Vec<Option<usize>> = vec![None; got.len()];

    fn try_augment(
        e: usize,
        adj: &[Vec<usize>],
        match_of_got: &mut [Option<usize>],
        visited: &mut [bool],
    ) -> bool {
        for &g in &adj[e] {
            if visited[g] {
                continue;
            }
            visited[g] = true;
            if match_of_got[g].is_none()
                || try_augment(match_of_got[g].unwrap(), adj, match_of_got, visited)
            {
                match_of_got[g] = Some(e);
                return true;
            }
        }
        false
    }

    for (e, expected_point) in expected.iter().enumerate().take(n) {
        let mut visited = vec![false; got.len()];
        if !try_augment(e, &adj, &mut match_of_got, &mut visited) {
            return Err(format!(
                "{label} missing expected point {expected_point:?}; got {got:?}"
            ));
        }
    }
    Ok(())
}

fn points_close(got: &[f64], expected: &[f64], atol: f64, rtol: f64) -> bool {
    got.len() == expected.len()
        && got
            .iter()
            .zip(expected.iter())
            .all(|(&g, &e)| allclose_scalar(g, e, atol, rtol))
}

/// Run the spatial conformance packet.
pub fn run_spatial_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: SpatialPacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute_spatial_case(case)
        })) {
            Ok(v) => v,
            Err(payload) => {
                case_results.push(CaseResult {
                    case_id: case.case_id().to_owned(),
                    passed: false,
                    message: format!(
                        "PANIC in execute_spatial_case: {}",
                        panic_payload_message(payload)
                    ),
                });
                continue;
            }
        };
        let (passed, message) = compare_spatial_outcome(case, &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

// ══════════════════════════════════════════════════════════════════════
// Signal Conformance Harness
// ══════════════════════════════════════════════════════════════════════

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalPacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<SignalCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalCase {
    pub case_id: String,
    pub category: String,
    pub mode: RuntimeMode,
    pub function: String,
    pub args: Vec<serde_json::Value>,
    pub expected: SignalExpected,
}

impl SignalCase {
    pub fn case_id(&self) -> &str {
        &self.case_id
    }
}

#[derive(Debug, Deserialize)]
pub struct SignalExpected {
    pub kind: String,
    pub value: Option<Vec<f64>>,
    pub b: Option<Vec<f64>>,
    pub a: Option<Vec<f64>>,
    pub w: Option<Vec<f64>>,
    pub h_mag: Option<Vec<f64>>,
    pub h_phase: Option<Vec<f64>>,
    pub real: Option<Vec<f64>>,
    pub imag: Option<Vec<f64>>,
    pub atol: Option<f64>,
    pub rtol: Option<f64>,
    #[serde(default)]
    pub contract_ref: String,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug)]
enum SignalObserved {
    Array(Vec<f64>),
    Indices(Vec<usize>),
    IirCoeffs {
        b: Vec<f64>,
        a: Vec<f64>,
    },
    Freqz {
        w: Vec<f64>,
        h_mag: Vec<f64>,
        h_phase: Vec<f64>,
    },
    ComplexArray {
        real: Vec<f64>,
        imag: Vec<f64>,
    },
    Error(String),
}

fn execute_signal_case(case: &SignalCase) -> SignalObserved {
    match case.function.as_str() {
        "savgol_filter" => execute_savgol_filter(case),
        "hann" | "hamming" | "blackman" | "boxcar" | "bartlett" | "flattop" | "cosine"
        | "blackmanharris" | "barthann" | "parzen" | "bohman" | "nuttall" | "lanczos" => {
            execute_window(case)
        }
        "kaiser" => execute_kaiser(case),
        "tukey" => execute_tukey(case),
        "gaussian" => execute_gaussian(case),
        "general_hamming" => execute_general_hamming(case),
        "exponential" => execute_exponential_window(case),
        "chebwin" => execute_chebwin(case),
        "kaiser_bessel_derived" => execute_kaiser_bessel_derived(case),
        "taylor" => execute_taylor(case),
        "general_cosine" => execute_general_cosine(case),
        "dpss" => execute_dpss(case),
        "convolve" => execute_convolve(case),
        "correlate" => execute_correlate(case),
        "find_peaks" => execute_find_peaks(case),
        "butter" | "cheby1" | "cheby2" | "ellip" | "bessel" => execute_signal_iir_design(case),
        "firwin" | "firwin2" | "firls" | "remez" | "minimum_phase" => {
            execute_signal_fir_design(case)
        }
        "freqz" => execute_freqz(case),
        "hilbert" => execute_hilbert(case),
        "detrend" => execute_detrend(case),
        _ => SignalObserved::Error(format!("unknown function: {}", case.function)),
    }
}

fn execute_savgol_filter(case: &SignalCase) -> SignalObserved {
    let x: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse x: {e}")),
    };
    let window_length: usize = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse window_length: {e}")),
    };
    let polyorder: usize = match serde_json::from_value(case.args[2].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse polyorder: {e}")),
    };
    match fsci_signal::savgol_filter(&x, window_length, polyorder) {
        Ok(result) => SignalObserved::Array(result),
        Err(e) => SignalObserved::Error(format!("{e:?}")),
    }
}

fn execute_window(case: &SignalCase) -> SignalObserved {
    let n: usize = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse n: {e}")),
    };
    let result = match case.function.as_str() {
        "hann" => fsci_signal::hann(n),
        "hamming" => fsci_signal::hamming(n),
        "blackman" => fsci_signal::blackman(n),
        // br-yz6i follow-up: extend window family.
        "boxcar" => fsci_signal::boxcar(n, true),
        "bartlett" => fsci_signal::bartlett(n),
        "flattop" => fsci_signal::flattop(n),
        "cosine" => fsci_signal::cosine(n),
        "blackmanharris" => fsci_signal::blackmanharris(n),
        "barthann" => fsci_signal::barthann(n),
        // br-drss: q6to slice 2 — 1-arg windows. nuttall is wired but
        // its fsci coefficients diverge from scipy's minimum-4-term
        // Blackman-Harris convention; not exercised in fixture today.
        "parzen" => fsci_signal::parzen(n),
        "bohman" => fsci_signal::bohman_window(n),
        "nuttall" => fsci_signal::nuttall_window(n),
        "lanczos" => fsci_signal::lanczos(n),
        _ => return SignalObserved::Error(format!("unknown window: {}", case.function)),
    };
    SignalObserved::Array(result)
}

fn execute_kaiser(case: &SignalCase) -> SignalObserved {
    let n: usize = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse n: {e}")),
    };
    let beta: f64 = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse beta: {e}")),
    };
    SignalObserved::Array(fsci_signal::kaiser(n, beta))
}

// br-z3ni: tukey window dispatch (m, alpha).
fn execute_tukey(case: &SignalCase) -> SignalObserved {
    let n: usize = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse n: {e}")),
    };
    let alpha: f64 = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse alpha: {e}")),
    };
    SignalObserved::Array(fsci_signal::tukey_window(n, alpha))
}

// br-z3ni: gaussian window dispatch (n, std). sym defaults to true.
fn execute_gaussian(case: &SignalCase) -> SignalObserved {
    let n: usize = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse n: {e}")),
    };
    let std: f64 = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse std: {e}")),
    };
    SignalObserved::Array(fsci_signal::gaussian(n, std, true))
}

// br-z3ni: general_hamming dispatch (n, alpha).
fn execute_general_hamming(case: &SignalCase) -> SignalObserved {
    let n: usize = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse n: {e}")),
    };
    let alpha: f64 = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse alpha: {e}")),
    };
    SignalObserved::Array(fsci_signal::general_hamming(n, alpha))
}

// br-z3ni: exponential window dispatch (n, tau). sym=true, center=None.
fn execute_exponential_window(case: &SignalCase) -> SignalObserved {
    let n: usize = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse n: {e}")),
    };
    let tau: f64 = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse tau: {e}")),
    };
    match fsci_signal::exponential(n, None, tau, true) {
        Ok(w) => SignalObserved::Array(w),
        Err(e) => SignalObserved::Error(format!("{e:?}")),
    }
}

// br-z3ni: chebwin (Dolph-Chebyshev) dispatch (n, at_dB).
fn execute_chebwin(case: &SignalCase) -> SignalObserved {
    let n: usize = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse n: {e}")),
    };
    let at: f64 = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse at: {e}")),
    };
    SignalObserved::Array(fsci_signal::chebwin(n, at))
}

// br-z3ni: kaiser_bessel_derived dispatch (n, beta). sym=true.
fn execute_kaiser_bessel_derived(case: &SignalCase) -> SignalObserved {
    let n: usize = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse n: {e}")),
    };
    let beta: f64 = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse beta: {e}")),
    };
    match fsci_signal::kaiser_bessel_derived(n, beta, true) {
        Ok(w) => SignalObserved::Array(w),
        Err(e) => SignalObserved::Error(format!("{e:?}")),
    }
}

// br-z3ni: taylor dispatch (n, nbar, sll, norm). sym defaults to true.
fn execute_taylor(case: &SignalCase) -> SignalObserved {
    let n: usize = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse n: {e}")),
    };
    let nbar: usize = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse nbar: {e}")),
    };
    let sll: f64 = match serde_json::from_value(case.args[2].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse sll: {e}")),
    };
    let norm: bool = match serde_json::from_value(case.args[3].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse norm: {e}")),
    };
    SignalObserved::Array(fsci_signal::taylor(n, nbar, sll, norm, true))
}

// br-z3ni: general_cosine dispatch (n, coeffs). sym=true.
fn execute_general_cosine(case: &SignalCase) -> SignalObserved {
    let n: usize = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse n: {e}")),
    };
    let coeffs: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse coeffs: {e}")),
    };
    SignalObserved::Array(fsci_signal::general_cosine(n, &coeffs, true))
}

// br-z3ni: dpss dispatch (n, nw, sym). kmax=None returns single taper at
// windows[0] with default Approximate normalization, matching scipy's
// scipy.signal.windows.dpss(M, NW, Kmax=None) symmetric default.
fn execute_dpss(case: &SignalCase) -> SignalObserved {
    let n: usize = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse n: {e}")),
    };
    let nw: f64 = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse nw: {e}")),
    };
    let sym: bool = match serde_json::from_value(case.args[2].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse sym: {e}")),
    };
    match fsci_signal::dpss(n, nw, None, sym, None, false) {
        Ok(result) => match result.windows.into_iter().next() {
            Some(window) => SignalObserved::Array(window),
            None => SignalObserved::Error("dpss returned no windows".to_string()),
        },
        Err(e) => SignalObserved::Error(format!("{e:?}")),
    }
}

fn execute_convolve(case: &SignalCase) -> SignalObserved {
    let a: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse a: {e}")),
    };
    let b: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse b: {e}")),
    };
    let mode: String = match serde_json::from_value(case.args[2].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse mode: {e}")),
    };
    let conv_mode = match mode.as_str() {
        "full" => fsci_signal::ConvolveMode::Full,
        "same" => fsci_signal::ConvolveMode::Same,
        "valid" => fsci_signal::ConvolveMode::Valid,
        _ => return SignalObserved::Error(format!("unknown mode: {mode}")),
    };
    match fsci_signal::convolve(&a, &b, conv_mode) {
        Ok(result) => SignalObserved::Array(result),
        Err(e) => SignalObserved::Error(format!("{e:?}")),
    }
}

fn execute_correlate(case: &SignalCase) -> SignalObserved {
    let a: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse a: {e}")),
    };
    let b: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse b: {e}")),
    };
    let mode: String = match serde_json::from_value(case.args[2].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse mode: {e}")),
    };
    // correlate uses ConvolveMode, same as convolve
    let corr_mode = match mode.as_str() {
        "full" => fsci_signal::ConvolveMode::Full,
        "same" => fsci_signal::ConvolveMode::Same,
        "valid" => fsci_signal::ConvolveMode::Valid,
        _ => return SignalObserved::Error(format!("unknown mode: {mode}")),
    };
    match fsci_signal::correlate(&a, &b, corr_mode) {
        Ok(result) => SignalObserved::Array(result),
        Err(e) => SignalObserved::Error(format!("{e:?}")),
    }
}

fn execute_find_peaks(case: &SignalCase) -> SignalObserved {
    let x: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse x: {e}")),
    };
    let options = fsci_signal::FindPeaksOptions {
        height: None,
        distance: None,
        prominence: None,
        width: None,
    };
    let result = fsci_signal::find_peaks(&x, options);
    SignalObserved::Indices(result.peaks)
}

fn parse_signal_f64(value: &serde_json::Value) -> Result<f64, String> {
    match value {
        serde_json::Value::Number(number) => number
            .as_f64()
            .ok_or_else(|| format!("number is not representable as f64: {number}")),
        serde_json::Value::String(value) => match value.to_ascii_lowercase().as_str() {
            "nan" => Ok(f64::NAN),
            "infinity" | "+infinity" | "inf" | "+inf" => Ok(f64::INFINITY),
            "-infinity" | "-inf" => Ok(f64::NEG_INFINITY),
            _ => value
                .parse::<f64>()
                .map_err(|e| format!("parse f64 string {value:?}: {e}")),
        },
        other => Err(format!("expected f64, got {other:?}")),
    }
}

fn parse_signal_wn(value: &serde_json::Value) -> Result<Vec<f64>, String> {
    match value {
        serde_json::Value::Array(values) => values.iter().map(parse_signal_f64).collect(),
        _ => parse_signal_f64(value).map(|single| vec![single]),
    }
}

fn parse_signal_filter_type(value: &serde_json::Value) -> Result<fsci_signal::FilterType, String> {
    let btype: String =
        serde_json::from_value(value.clone()).map_err(|e| format!("parse btype: {e}"))?;
    match btype.as_str() {
        "low" | "lowpass" => Ok(fsci_signal::FilterType::Lowpass),
        "high" | "highpass" => Ok(fsci_signal::FilterType::Highpass),
        "band" | "bandpass" => Ok(fsci_signal::FilterType::Bandpass),
        "stop" | "bandstop" => Ok(fsci_signal::FilterType::Bandstop),
        _ => Err(format!("unknown filter type: {btype}")),
    }
}

fn signal_iir_coeffs(
    result: Result<fsci_signal::BaCoeffs, fsci_signal::SignalError>,
) -> SignalObserved {
    match result {
        Ok(coeffs) => SignalObserved::IirCoeffs {
            b: coeffs.b,
            a: coeffs.a,
        },
        Err(e) => SignalObserved::Error(format!("{e:?}")),
    }
}

fn execute_signal_iir_design(case: &SignalCase) -> SignalObserved {
    let order: usize = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse order: {e}")),
    };
    let parsed = match case.function.as_str() {
        "butter" | "bessel" => {
            let wn = parse_signal_wn(&case.args[1]);
            let filter_type = case.args.get(2).map_or_else(
                || Ok(fsci_signal::FilterType::Lowpass),
                parse_signal_filter_type,
            );
            wn.and_then(|wn| filter_type.map(|filter_type| (wn, None, None, filter_type)))
        }
        "cheby1" => {
            let rp = parse_signal_f64(&case.args[1]);
            let wn = parse_signal_wn(&case.args[2]);
            let filter_type = case.args.get(3).map_or_else(
                || Ok(fsci_signal::FilterType::Lowpass),
                parse_signal_filter_type,
            );
            rp.and_then(|rp| {
                wn.and_then(|wn| filter_type.map(|filter_type| (wn, Some(rp), None, filter_type)))
            })
        }
        "cheby2" => {
            let rs = parse_signal_f64(&case.args[1]);
            let wn = parse_signal_wn(&case.args[2]);
            let filter_type = case.args.get(3).map_or_else(
                || Ok(fsci_signal::FilterType::Lowpass),
                parse_signal_filter_type,
            );
            rs.and_then(|rs| {
                wn.and_then(|wn| filter_type.map(|filter_type| (wn, None, Some(rs), filter_type)))
            })
        }
        "ellip" => {
            let rp = parse_signal_f64(&case.args[1]);
            let rs = parse_signal_f64(&case.args[2]);
            let wn = parse_signal_wn(&case.args[3]);
            let filter_type = case.args.get(4).map_or_else(
                || Ok(fsci_signal::FilterType::Lowpass),
                parse_signal_filter_type,
            );
            rp.and_then(|rp| {
                rs.and_then(|rs| {
                    wn.and_then(|wn| {
                        filter_type.map(|filter_type| (wn, Some(rp), Some(rs), filter_type))
                    })
                })
            })
        }
        _ => return SignalObserved::Error(format!("unknown IIR design: {}", case.function)),
    };
    let (wn, rp, rs, filter_type) = match parsed {
        Ok(parsed) => parsed,
        Err(e) => return SignalObserved::Error(e),
    };

    match case.function.as_str() {
        "butter" => signal_iir_coeffs(fsci_signal::butter(order, &wn, filter_type)),
        "cheby1" => signal_iir_coeffs(fsci_signal::cheby1(
            order,
            rp.unwrap_or(1.0),
            &wn,
            filter_type,
        )),
        "cheby2" => signal_iir_coeffs(fsci_signal::cheby2(
            order,
            rs.unwrap_or(20.0),
            &wn,
            filter_type,
        )),
        "ellip" => signal_iir_coeffs(fsci_signal::ellip(
            order,
            rp.unwrap_or(1.0),
            rs.unwrap_or(40.0),
            &wn,
            filter_type,
        )),
        "bessel" => signal_iir_coeffs(fsci_signal::bessel(order, &wn, filter_type)),
        _ => SignalObserved::Error(format!("unknown IIR design: {}", case.function)),
    }
}

/// br-7jrx: dispatch FIR design functions (firwin / firwin2 / firls /
/// remez / minimum_phase). Each case's args layout:
///
///   firwin:        [numtaps, cutoff (scalar or array), window, pass_zero]
///   firwin2:       [numtaps, freq[], gain[], window]
///   firls:         [numtaps, bands[], desired[]]   (weight optional via 4th arg)
///   remez:         [numtaps, bands[], desired[]]   (weight optional via 4th arg)
///   minimum_phase: [h[]]  (FIR taps of a linear-phase prototype)
fn execute_signal_fir_design(case: &SignalCase) -> SignalObserved {
    fn parse_window(value: Option<&serde_json::Value>) -> fsci_signal::FirWindow {
        match value
            .and_then(serde_json::Value::as_str)
            .unwrap_or("hamming")
        {
            "hann" => fsci_signal::FirWindow::Hann,
            "blackman" => fsci_signal::FirWindow::Blackman,
            "rectangular" | "boxcar" | "rect" => fsci_signal::FirWindow::Rectangular,
            other => {
                if let Some(beta_str) = other.strip_prefix("kaiser:")
                    && let Ok(beta) = beta_str.parse::<f64>()
                {
                    return fsci_signal::FirWindow::Kaiser(beta);
                }
                fsci_signal::FirWindow::Hamming
            }
        }
    }
    fn vec_f64(value: &serde_json::Value) -> Result<Vec<f64>, String> {
        serde_json::from_value(value.clone()).map_err(|e| format!("parse vec: {e}"))
    }
    fn scalar_or_vec_f64(value: &serde_json::Value) -> Result<Vec<f64>, String> {
        if let Some(x) = value.as_f64() {
            return Ok(vec![x]);
        }
        vec_f64(value)
    }

    match case.function.as_str() {
        "firwin" => {
            let numtaps: usize = match serde_json::from_value(case.args[0].clone()) {
                Ok(v) => v,
                Err(e) => return SignalObserved::Error(format!("parse numtaps: {e}")),
            };
            let cutoff = match scalar_or_vec_f64(&case.args[1]) {
                Ok(v) => v,
                Err(e) => return SignalObserved::Error(format!("parse cutoff: {e}")),
            };
            let window = parse_window(case.args.get(2));
            let pass_zero = case
                .args
                .get(3)
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true);
            match fsci_signal::firwin(numtaps, &cutoff, window, pass_zero) {
                Ok(h) => SignalObserved::Array(h),
                Err(e) => SignalObserved::Error(format!("{e:?}")),
            }
        }
        "firwin2" => {
            let numtaps: usize = match serde_json::from_value(case.args[0].clone()) {
                Ok(v) => v,
                Err(e) => return SignalObserved::Error(format!("parse numtaps: {e}")),
            };
            let freq = match vec_f64(&case.args[1]) {
                Ok(v) => v,
                Err(e) => return SignalObserved::Error(e),
            };
            let gain = match vec_f64(&case.args[2]) {
                Ok(v) => v,
                Err(e) => return SignalObserved::Error(e),
            };
            let window = parse_window(case.args.get(3));
            match fsci_signal::firwin2(numtaps, &freq, &gain, window) {
                Ok(h) => SignalObserved::Array(h),
                Err(e) => SignalObserved::Error(format!("{e:?}")),
            }
        }
        "firls" => {
            let numtaps: usize = match serde_json::from_value(case.args[0].clone()) {
                Ok(v) => v,
                Err(e) => return SignalObserved::Error(format!("parse numtaps: {e}")),
            };
            let bands = match vec_f64(&case.args[1]) {
                Ok(v) => v,
                Err(e) => return SignalObserved::Error(e),
            };
            let desired = match vec_f64(&case.args[2]) {
                Ok(v) => v,
                Err(e) => return SignalObserved::Error(e),
            };
            let weight = case.args.get(3).and_then(|v| vec_f64(v).ok());
            match fsci_signal::firls(numtaps, &bands, &desired, weight.as_deref()) {
                Ok(h) => SignalObserved::Array(h),
                Err(e) => SignalObserved::Error(format!("{e:?}")),
            }
        }
        "remez" => {
            let numtaps: usize = match serde_json::from_value(case.args[0].clone()) {
                Ok(v) => v,
                Err(e) => return SignalObserved::Error(format!("parse numtaps: {e}")),
            };
            let bands = match vec_f64(&case.args[1]) {
                Ok(v) => v,
                Err(e) => return SignalObserved::Error(e),
            };
            let desired = match vec_f64(&case.args[2]) {
                Ok(v) => v,
                Err(e) => return SignalObserved::Error(e),
            };
            let weight = case.args.get(3).and_then(|v| vec_f64(v).ok());
            match fsci_signal::remez(numtaps, &bands, &desired, weight.as_deref()) {
                Ok(h) => SignalObserved::Array(h),
                Err(e) => SignalObserved::Error(format!("{e:?}")),
            }
        }
        "minimum_phase" => {
            let h = match vec_f64(&case.args[0]) {
                Ok(v) => v,
                Err(e) => return SignalObserved::Error(e),
            };
            match fsci_signal::minimum_phase(&h) {
                Ok(h) => SignalObserved::Array(h),
                Err(e) => SignalObserved::Error(format!("{e:?}")),
            }
        }
        _ => SignalObserved::Error(format!("unknown FIR design: {}", case.function)),
    }
}

fn execute_freqz(case: &SignalCase) -> SignalObserved {
    let b: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse b: {e}")),
    };
    let a: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse a: {e}")),
    };
    let wor_n: usize = match serde_json::from_value(case.args[2].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse worN: {e}")),
    };
    match fsci_signal::freqz(&b, &a, Some(wor_n)) {
        Ok(result) => SignalObserved::Freqz {
            w: result.w,
            h_mag: result.h_mag,
            h_phase: result.h_phase,
        },
        Err(e) => SignalObserved::Error(format!("{e:?}")),
    }
}

fn execute_hilbert(case: &SignalCase) -> SignalObserved {
    let x: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse x: {e}")),
    };
    match fsci_signal::hilbert(&x) {
        Ok(result) => SignalObserved::ComplexArray {
            real: result.iter().map(|c| c.0).collect(),
            imag: result.iter().map(|c| c.1).collect(),
        },
        Err(e) => SignalObserved::Error(format!("{e:?}")),
    }
}

fn execute_detrend(case: &SignalCase) -> SignalObserved {
    let x: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse x: {e}")),
    };
    let dtype: String = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return SignalObserved::Error(format!("parse type: {e}")),
    };
    let detrend_type = match dtype.as_str() {
        "linear" => fsci_signal::DetrendType::Linear,
        "constant" => fsci_signal::DetrendType::Constant,
        _ => return SignalObserved::Error(format!("unknown detrend type: {dtype}")),
    };
    match fsci_signal::detrend(&x, detrend_type) {
        Ok(result) => SignalObserved::Array(result),
        Err(e) => SignalObserved::Error(format!("{e:?}")),
    }
}

fn compare_signal_iir_coefficients(
    case: &SignalCase,
    b: &[f64],
    a: &[f64],
    atol: f64,
    rtol: f64,
) -> (bool, String) {
    let exp_b = case.expected.b.as_ref().cloned().unwrap_or_default();
    let exp_a = case.expected.a.as_ref().cloned().unwrap_or_default();
    if b.len() != exp_b.len() || a.len() != exp_a.len() {
        return (
            false,
            format!(
                "length mismatch: b({}/{}), a({}/{})",
                b.len(),
                exp_b.len(),
                a.len(),
                exp_a.len()
            ),
        );
    }
    for (i, (&g, &e)) in b.iter().zip(exp_b.iter()).enumerate() {
        if !allclose_scalar(g, e, atol, rtol) {
            return (false, format!("b[{i}] mismatch: got {g}, expected {e}"));
        }
    }
    for (i, (&g, &e)) in a.iter().zip(exp_a.iter()).enumerate() {
        if !allclose_scalar(g, e, atol, rtol) {
            return (false, format!("a[{i}] mismatch: got {g}, expected {e}"));
        }
    }
    (true, "iir coeffs match".to_string())
}

fn compare_signal_outcome(case: &SignalCase, observed: &SignalObserved) -> (bool, String) {
    let atol = case.expected.atol.unwrap_or(1e-10);
    let rtol = case.expected.rtol.unwrap_or(1e-10);

    match (case.expected.kind.as_str(), observed) {
        ("array", SignalObserved::Array(got)) => {
            let expected = case.expected.value.as_ref().cloned().unwrap_or_default();
            if got.len() != expected.len() {
                return (
                    false,
                    format!(
                        "length mismatch: got {}, expected {}",
                        got.len(),
                        expected.len()
                    ),
                );
            }
            for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
                if !allclose_scalar(g, e, atol, rtol) {
                    return (false, format!("array[{i}] mismatch: got {g}, expected {e}"));
                }
            }
            (true, format!("array match ({} elements)", got.len()))
        }
        ("indices", SignalObserved::Indices(got)) => {
            let expected: Vec<usize> = case
                .expected
                .value
                .as_ref()
                .map(|v| v.iter().map(|&x| x as usize).collect())
                .unwrap_or_default();
            if *got == expected {
                (true, format!("indices match: {got:?}"))
            } else {
                (
                    false,
                    format!("indices mismatch: got {got:?}, expected {expected:?}"),
                )
            }
        }
        ("iir_coeffs", SignalObserved::IirCoeffs { b, a }) => {
            compare_signal_iir_coefficients(case, b, a, atol, rtol)
        }
        ("freqz_result", SignalObserved::Freqz { w, h_mag, h_phase }) => {
            let exp_w = case.expected.w.as_ref().cloned().unwrap_or_default();
            let exp_h_mag = case.expected.h_mag.as_ref().cloned().unwrap_or_default();
            let exp_h_phase = case.expected.h_phase.as_ref().cloned().unwrap_or_default();
            if w.len() != exp_w.len() {
                return (
                    false,
                    format!("w length mismatch: {}/{}", w.len(), exp_w.len()),
                );
            }
            for (i, (&g, &e)) in w.iter().zip(exp_w.iter()).enumerate() {
                if !allclose_scalar(g, e, atol, rtol) {
                    return (false, format!("w[{i}] mismatch: got {g}, expected {e}"));
                }
            }
            for (i, (&g, &e)) in h_mag.iter().zip(exp_h_mag.iter()).enumerate() {
                if !allclose_scalar(g, e, atol, rtol) {
                    return (false, format!("h_mag[{i}] mismatch: got {g}, expected {e}"));
                }
            }
            for (i, (&g, &e)) in h_phase.iter().zip(exp_h_phase.iter()).enumerate() {
                if !allclose_scalar(g, e, atol, rtol) {
                    return (
                        false,
                        format!("h_phase[{i}] mismatch: got {g}, expected {e}"),
                    );
                }
            }
            (true, format!("freqz match ({} points)", w.len()))
        }
        ("complex_array", SignalObserved::ComplexArray { real, imag }) => {
            let exp_real = case.expected.real.as_ref().cloned().unwrap_or_default();
            let exp_imag = case.expected.imag.as_ref().cloned().unwrap_or_default();
            if real.len() != exp_real.len() {
                return (
                    false,
                    format!("real length mismatch: {}/{}", real.len(), exp_real.len()),
                );
            }
            for (i, (&g, &e)) in real.iter().zip(exp_real.iter()).enumerate() {
                if !allclose_scalar(g, e, atol, rtol) {
                    return (false, format!("real[{i}] mismatch: got {g}, expected {e}"));
                }
            }
            for (i, (&g, &e)) in imag.iter().zip(exp_imag.iter()).enumerate() {
                if !allclose_scalar(g, e, atol, rtol) {
                    return (false, format!("imag[{i}] mismatch: got {g}, expected {e}"));
                }
            }
            (
                true,
                format!("complex array match ({} elements)", real.len()),
            )
        }
        ("error", SignalObserved::Error(actual))
        | ("error_kind", SignalObserved::Error(actual)) => {
            let expected = case.expected.error.as_deref().unwrap_or_default();
            if matches_error_contract(actual, expected) {
                (true, format!("error matched: {actual}"))
            } else {
                (
                    false,
                    format!("error mismatch: got {actual}, expected {expected}"),
                )
            }
        }
        (_, SignalObserved::Error(e)) => (false, format!("execution error: {e}")),
        (kind, obs) => (
            false,
            format!("type mismatch: expected {kind}, got {obs:?}"),
        ),
    }
}

/// Run the signal conformance packet.
pub fn run_signal_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: SignalPacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute_signal_case(case)
        })) {
            Ok(v) => v,
            Err(payload) => {
                case_results.push(CaseResult {
                    case_id: case.case_id().to_owned(),
                    passed: false,
                    message: format!(
                        "PANIC in execute_signal_case: {}",
                        panic_payload_message(payload)
                    ),
                });
                continue;
            }
        };
        let (passed, message) = compare_signal_outcome(case, &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

// ══════════════════════════════════════════════════════════════════════
// Stats Conformance Harness
// ══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatsPacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<StatsCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatsCase {
    pub case_id: String,
    pub category: String,
    pub mode: RuntimeMode,
    pub function: String,
    pub args: Vec<serde_json::Value>,
    pub expected: StatsExpected,
}

impl StatsCase {
    pub fn case_id(&self) -> &str {
        &self.case_id
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsExpected {
    pub kind: String,
    pub value: Option<f64>,
    // describe_result fields
    pub nobs: Option<usize>,
    pub minmax: Option<[f64; 2]>,
    pub mean: Option<f64>,
    pub variance: Option<f64>,
    pub skewness: Option<f64>,
    pub kurtosis: Option<f64>,
    // correlation_result fields
    pub statistic: Option<f64>,
    pub pvalue: Option<f64>,
    // linregress_result fields
    pub slope: Option<f64>,
    pub intercept: Option<f64>,
    pub rvalue: Option<f64>,
    pub stderr: Option<f64>,
    // array output
    #[serde(default)]
    pub array_value: Option<Vec<f64>>,
    pub atol: Option<f64>,
    pub rtol: Option<f64>,
    #[serde(default)]
    pub contract_ref: String,
}

#[derive(Debug)]
enum StatsObserved {
    Scalar(f64),
    Array(Vec<f64>),
    Describe {
        nobs: usize,
        minmax: [f64; 2],
        mean: f64,
        variance: f64,
        skewness: f64,
        kurtosis: f64,
    },
    Correlation {
        statistic: f64,
        pvalue: f64,
    },
    Linregress {
        slope: f64,
        intercept: f64,
        rvalue: f64,
        pvalue: f64,
        stderr: f64,
    },
    Ttest {
        statistic: f64,
        pvalue: f64,
    },
    Goodness {
        statistic: f64,
        pvalue: f64,
    },
    Error(String),
}

fn execute_stats_case(case: &StatsCase) -> StatsObserved {
    match case.function.as_str() {
        "describe" => execute_stats_describe(case),
        "skew" => execute_stats_skew(case),
        "kurtosis" => execute_stats_kurtosis(case),
        "pearsonr" => execute_stats_pearsonr(case),
        "spearmanr" => execute_stats_spearmanr(case),
        "kendalltau" => execute_stats_kendalltau(case),
        "linregress" => execute_stats_linregress(case),
        "ttest_1samp" => execute_stats_ttest_1samp(case),
        "ttest_ind" => execute_stats_ttest_ind(case),
        "mannwhitneyu" => execute_stats_mannwhitneyu(case),
        "wilcoxon" => execute_stats_wilcoxon(case),
        "ks_2samp" => execute_stats_ks_2samp(case),
        "zscore" => execute_stats_zscore(case),
        "sem" => execute_stats_sem(case),
        "iqr" => execute_stats_iqr(case),
        "moment" => execute_stats_moment(case),
        "variation" => execute_stats_variation(case),
        "entropy" => execute_stats_entropy(case),
        "shapiro" => execute_stats_shapiro(case),
        "distribution_pdf" => execute_stats_distribution_pdf(case),
        "distribution_cdf" => execute_stats_distribution_cdf(case),
        "distribution_ppf" => execute_stats_distribution_ppf(case),
        "bootstrap_mean" => execute_stats_bootstrap_mean(case),
        _ => StatsObserved::Error(format!("unknown function: {}", case.function)),
    }
}

fn execute_stats_describe(case: &StatsCase) -> StatsObserved {
    let data: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse data: {e}")),
    };
    let result = fsci_stats::describe(&data);
    StatsObserved::Describe {
        nobs: result.nobs,
        minmax: [result.minmax.0, result.minmax.1],
        mean: result.mean,
        variance: result.variance,
        skewness: result.skewness,
        kurtosis: result.kurtosis,
    }
}

fn execute_stats_skew(case: &StatsCase) -> StatsObserved {
    let data: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse data: {e}")),
    };
    StatsObserved::Scalar(fsci_stats::skew(&data))
}

fn execute_stats_kurtosis(case: &StatsCase) -> StatsObserved {
    let data: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse data: {e}")),
    };
    StatsObserved::Scalar(fsci_stats::kurtosis(&data))
}

fn execute_stats_pearsonr(case: &StatsCase) -> StatsObserved {
    let x: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse x: {e}")),
    };
    let y: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse y: {e}")),
    };
    let result = fsci_stats::pearsonr(&x, &y);
    StatsObserved::Correlation {
        statistic: result.statistic,
        pvalue: result.pvalue,
    }
}

fn execute_stats_spearmanr(case: &StatsCase) -> StatsObserved {
    let x: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse x: {e}")),
    };
    let y: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse y: {e}")),
    };
    let result = fsci_stats::spearmanr(&x, &y);
    StatsObserved::Correlation {
        statistic: result.statistic,
        pvalue: result.pvalue,
    }
}

// br-q00y: Kendall tau correlation parity dispatch.
fn execute_stats_kendalltau(case: &StatsCase) -> StatsObserved {
    let x: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse x: {e}")),
    };
    let y: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse y: {e}")),
    };
    let result = fsci_stats::kendalltau(&x, &y);
    StatsObserved::Correlation {
        statistic: result.statistic,
        pvalue: result.pvalue,
    }
}

fn execute_stats_linregress(case: &StatsCase) -> StatsObserved {
    let x: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse x: {e}")),
    };
    let y: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse y: {e}")),
    };
    let result = fsci_stats::linregress(&x, &y);
    StatsObserved::Linregress {
        slope: result.slope,
        intercept: result.intercept,
        rvalue: result.rvalue,
        pvalue: result.pvalue,
        stderr: result.stderr,
    }
}

fn execute_stats_ttest_1samp(case: &StatsCase) -> StatsObserved {
    let data: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse data: {e}")),
    };
    let popmean: f64 = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse popmean: {e}")),
    };
    let result = fsci_stats::ttest_1samp(&data, popmean);
    StatsObserved::Ttest {
        statistic: result.statistic,
        pvalue: result.pvalue,
    }
}

// br-7k5n: 2-sample location tests routing through fsci_stats. All
// three return TtestResult { statistic, pvalue }; the conformance
// comparator already handles "ttest_result" expected.kind.
fn execute_stats_ttest_ind(case: &StatsCase) -> StatsObserved {
    let a: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse a: {e}")),
    };
    let b: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse b: {e}")),
    };
    let result = fsci_stats::ttest_ind(&a, &b);
    StatsObserved::Ttest {
        statistic: result.statistic,
        pvalue: result.pvalue,
    }
}

fn execute_stats_mannwhitneyu(case: &StatsCase) -> StatsObserved {
    let x: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse x: {e}")),
    };
    let y: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse y: {e}")),
    };
    let result = fsci_stats::mannwhitneyu(&x, &y);
    StatsObserved::Ttest {
        statistic: result.statistic,
        pvalue: result.pvalue,
    }
}

fn execute_stats_wilcoxon(case: &StatsCase) -> StatsObserved {
    let x: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse x: {e}")),
    };
    let y: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse y: {e}")),
    };
    let result = fsci_stats::wilcoxon(&x, &y);
    StatsObserved::Ttest {
        statistic: result.statistic,
        pvalue: result.pvalue,
    }
}

fn execute_stats_ks_2samp(case: &StatsCase) -> StatsObserved {
    let data1: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse data1: {e}")),
    };
    let data2: Vec<f64> = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse data2: {e}")),
    };
    let result = fsci_stats::ks_2samp(&data1, &data2);
    StatsObserved::Goodness {
        statistic: result.statistic,
        pvalue: result.pvalue,
    }
}

fn execute_stats_zscore(case: &StatsCase) -> StatsObserved {
    let data: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse data: {e}")),
    };
    StatsObserved::Array(fsci_stats::zscore(&data))
}

fn execute_stats_sem(case: &StatsCase) -> StatsObserved {
    let data: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse data: {e}")),
    };
    StatsObserved::Scalar(fsci_stats::sem(&data))
}

fn execute_stats_iqr(case: &StatsCase) -> StatsObserved {
    let data: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse data: {e}")),
    };
    StatsObserved::Scalar(fsci_stats::iqr(&data))
}

fn execute_stats_moment(case: &StatsCase) -> StatsObserved {
    let data: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse data: {e}")),
    };
    let k: u32 = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse k: {e}")),
    };
    StatsObserved::Scalar(fsci_stats::moment(&data, k))
}

fn execute_stats_variation(case: &StatsCase) -> StatsObserved {
    let data: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse data: {e}")),
    };
    StatsObserved::Scalar(fsci_stats::variation(&data))
}

fn execute_stats_entropy(case: &StatsCase) -> StatsObserved {
    let pk: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse pk: {e}")),
    };
    StatsObserved::Scalar(fsci_stats::entropy(&pk, None))
}

fn execute_stats_shapiro(case: &StatsCase) -> StatsObserved {
    let data: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse data: {e}")),
    };
    let result = fsci_stats::shapiro(&data);
    StatsObserved::Goodness {
        statistic: result.statistic,
        pvalue: result.pvalue,
    }
}

// br-4nkk: continuous-distribution pdf/cdf/ppf parity dispatch.
//
// Schema: args = [distribution_name: str, params: [f64...], x: f64].
// Inline dispatch (no Box<dyn>) since ContinuousDistribution has a
// generic `rvs<R: Rng>` method making it not dyn-compatible.
fn parse_distribution_args(case: &StatsCase) -> Result<(String, Vec<f64>, f64), String> {
    let name: String = serde_json::from_value(case.args[0].clone())
        .map_err(|e| format!("parse distribution name: {e}"))?;
    let params: Vec<f64> =
        serde_json::from_value(case.args[1].clone()).map_err(|e| format!("parse params: {e}"))?;
    let x: f64 =
        serde_json::from_value(case.args[2].clone()).map_err(|e| format!("parse x: {e}"))?;
    Ok((name, params, x))
}

fn evaluate_distribution_method(
    name: &str,
    params: &[f64],
    x: f64,
    method: &str,
) -> Result<f64, String> {
    use fsci_stats::ContinuousDistribution;
    match name {
        "Cauchy" => {
            if params.len() != 2 {
                return Err(format!("Cauchy requires (loc, scale), got {params:?}"));
            }
            let d = fsci_stats::Cauchy {
                loc: params[0],
                scale: params[1],
            };
            Ok(match method {
                "pdf" => d.pdf(x),
                "cdf" => d.cdf(x),
                "ppf" => d.ppf(x),
                _ => return Err(format!("unsupported method: {method}")),
            })
        }
        "ChiSquared" => {
            if params.len() != 1 {
                return Err(format!("ChiSquared requires (df,), got {params:?}"));
            }
            let d = fsci_stats::ChiSquared { df: params[0] };
            Ok(match method {
                "pdf" => d.pdf(x),
                "cdf" => d.cdf(x),
                "ppf" => d.ppf(x),
                _ => return Err(format!("unsupported method: {method}")),
            })
        }
        "Lognormal" => {
            if params.len() != 2 {
                return Err(format!("Lognormal requires (s, scale), got {params:?}"));
            }
            let d = fsci_stats::Lognormal {
                s: params[0],
                scale: params[1],
            };
            Ok(match method {
                "pdf" => d.pdf(x),
                "cdf" => d.cdf(x),
                "ppf" => d.ppf(x),
                _ => return Err(format!("unsupported method: {method}")),
            })
        }
        "GammaDist" => {
            if params.len() != 2 {
                return Err(format!("GammaDist requires (a, scale), got {params:?}"));
            }
            let d = fsci_stats::GammaDist {
                a: params[0],
                scale: params[1],
            };
            Ok(match method {
                "pdf" => d.pdf(x),
                "cdf" => d.cdf(x),
                "ppf" => d.ppf(x),
                _ => return Err(format!("unsupported method: {method}")),
            })
        }
        // br-5sra (4nkk slice 3): 5 more distributions.
        "BetaDist" => {
            if params.len() != 2 {
                return Err(format!("BetaDist requires (a, b), got {params:?}"));
            }
            let d = fsci_stats::BetaDist {
                a: params[0],
                b: params[1],
            };
            Ok(match method {
                "pdf" => d.pdf(x),
                "cdf" => d.cdf(x),
                "ppf" => d.ppf(x),
                _ => return Err(format!("unsupported method: {method}")),
            })
        }
        "Pareto" => {
            if params.len() != 2 {
                return Err(format!("Pareto requires (b, scale), got {params:?}"));
            }
            let d = fsci_stats::Pareto {
                b: params[0],
                scale: params[1],
            };
            Ok(match method {
                "pdf" => d.pdf(x),
                "cdf" => d.cdf(x),
                "ppf" => d.ppf(x),
                _ => return Err(format!("unsupported method: {method}")),
            })
        }
        "Rayleigh" => {
            if params.len() != 1 {
                return Err(format!("Rayleigh requires (scale,), got {params:?}"));
            }
            let d = fsci_stats::Rayleigh { scale: params[0] };
            Ok(match method {
                "pdf" => d.pdf(x),
                "cdf" => d.cdf(x),
                "ppf" => d.ppf(x),
                _ => return Err(format!("unsupported method: {method}")),
            })
        }
        "Logistic" => {
            if params.len() != 2 {
                return Err(format!("Logistic requires (loc, scale), got {params:?}"));
            }
            let d = fsci_stats::Logistic {
                loc: params[0],
                scale: params[1],
            };
            Ok(match method {
                "pdf" => d.pdf(x),
                "cdf" => d.cdf(x),
                "ppf" => d.ppf(x),
                _ => return Err(format!("unsupported method: {method}")),
            })
        }
        "Laplace" => {
            if params.len() != 2 {
                return Err(format!("Laplace requires (loc, scale), got {params:?}"));
            }
            let d = fsci_stats::Laplace {
                loc: params[0],
                scale: params[1],
            };
            Ok(match method {
                "pdf" => d.pdf(x),
                "cdf" => d.cdf(x),
                "ppf" => d.ppf(x),
                _ => return Err(format!("unsupported method: {method}")),
            })
        }
        // br-a6qx slice 2 deferral closure: F-distribution
        "FDistribution" => {
            if params.len() != 2 {
                return Err(format!("FDistribution requires (dfn, dfd), got {params:?}"));
            }
            let d = fsci_stats::FDistribution {
                dfn: params[0],
                dfd: params[1],
            };
            Ok(match method {
                "pdf" => d.pdf(x),
                "cdf" => d.cdf(x),
                "ppf" => d.ppf(x),
                _ => return Err(format!("unsupported method: {method}")),
            })
        }
        other => Err(format!("unsupported distribution: {other}")),
    }
}

fn execute_stats_distribution_pdf(case: &StatsCase) -> StatsObserved {
    let (name, params, x) = match parse_distribution_args(case) {
        Ok(t) => t,
        Err(e) => return StatsObserved::Error(e),
    };
    match evaluate_distribution_method(&name, &params, x, "pdf") {
        Ok(v) => StatsObserved::Scalar(v),
        Err(e) => StatsObserved::Error(e),
    }
}

fn execute_stats_distribution_cdf(case: &StatsCase) -> StatsObserved {
    let (name, params, x) = match parse_distribution_args(case) {
        Ok(t) => t,
        Err(e) => return StatsObserved::Error(e),
    };
    match evaluate_distribution_method(&name, &params, x, "cdf") {
        Ok(v) => StatsObserved::Scalar(v),
        Err(e) => StatsObserved::Error(e),
    }
}

fn execute_stats_distribution_ppf(case: &StatsCase) -> StatsObserved {
    let (name, params, x) = match parse_distribution_args(case) {
        Ok(t) => t,
        Err(e) => return StatsObserved::Error(e),
    };
    match evaluate_distribution_method(&name, &params, x, "ppf") {
        Ok(v) => StatsObserved::Scalar(v),
        Err(e) => StatsObserved::Error(e),
    }
}

// br-7vhu: bootstrap_mean dispatch. Args: (data, n_bootstrap, confidence,
// seed). Returns the BCa CI bounds as [lo, hi]. Expected fixtures lock
// fsci's deterministic LCG output at fixed seed; scipy parity is
// blocked on PRNG mismatch (scipy uses PCG64) and tracked separately.
fn execute_stats_bootstrap_mean(case: &StatsCase) -> StatsObserved {
    let data: Vec<f64> = match serde_json::from_value(case.args[0].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse data: {e}")),
    };
    let n_bootstrap: usize = match serde_json::from_value(case.args[1].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse n_bootstrap: {e}")),
    };
    let confidence: f64 = match serde_json::from_value(case.args[2].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse confidence: {e}")),
    };
    let seed: u64 = match serde_json::from_value(case.args[3].clone()) {
        Ok(v) => v,
        Err(e) => return StatsObserved::Error(format!("parse seed: {e}")),
    };
    let (lo, hi) = fsci_stats::bootstrap_mean(&data, n_bootstrap, confidence, seed);
    StatsObserved::Array(vec![lo, hi])
}

fn compare_stats_outcome(case: &StatsCase, observed: &StatsObserved) -> (bool, String) {
    let atol = case.expected.atol.unwrap_or(1e-10);
    let rtol = case.expected.rtol.unwrap_or(1e-10);

    fn close(a: f64, b: f64, atol: f64, rtol: f64) -> bool {
        allclose_scalar(a, b, atol, rtol)
    }

    match (case.expected.kind.as_str(), observed) {
        ("scalar", StatsObserved::Scalar(got)) => {
            let expected = case.expected.value.unwrap_or(0.0);
            if close(*got, expected, atol, rtol) {
                (true, format!("scalar match: {got}"))
            } else {
                (
                    false,
                    format!("scalar mismatch: got {got}, expected {expected}"),
                )
            }
        }
        ("array", StatsObserved::Array(got)) => {
            // Try array_value first, fall back to trying value as Option<Vec<f64>>
            let expected: Vec<f64> = case.expected.array_value.clone().unwrap_or_default();
            if got.len() != expected.len() {
                return (
                    false,
                    format!(
                        "length mismatch: got {}, expected {}",
                        got.len(),
                        expected.len()
                    ),
                );
            }
            for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
                if !close(g, e, atol, rtol) {
                    return (false, format!("array[{i}] mismatch: got {g}, expected {e}"));
                }
            }
            (true, format!("array match ({} elements)", got.len()))
        }
        (
            "describe_result",
            StatsObserved::Describe {
                nobs,
                minmax,
                mean,
                variance,
                skewness,
                kurtosis,
            },
        ) => {
            let exp_nobs = case.expected.nobs.unwrap_or(0);
            let exp_minmax = case.expected.minmax.unwrap_or([0.0, 0.0]);
            let exp_mean = case.expected.mean.unwrap_or(0.0);
            let exp_variance = case.expected.variance.unwrap_or(0.0);
            let exp_skewness = case.expected.skewness.unwrap_or(0.0);
            let exp_kurtosis = case.expected.kurtosis.unwrap_or(0.0);

            if *nobs != exp_nobs {
                return (
                    false,
                    format!("nobs mismatch: got {nobs}, expected {exp_nobs}"),
                );
            }
            if !close(minmax[0], exp_minmax[0], atol, rtol)
                || !close(minmax[1], exp_minmax[1], atol, rtol)
            {
                return (
                    false,
                    format!(
                        "minmax mismatch: got {:?}, expected {:?}",
                        minmax, exp_minmax
                    ),
                );
            }
            if !close(*mean, exp_mean, atol, rtol) {
                return (
                    false,
                    format!("mean mismatch: got {mean}, expected {exp_mean}"),
                );
            }
            if !close(*variance, exp_variance, atol, rtol) {
                return (
                    false,
                    format!("variance mismatch: got {variance}, expected {exp_variance}"),
                );
            }
            if !close(*skewness, exp_skewness, atol, rtol) {
                return (
                    false,
                    format!("skewness mismatch: got {skewness}, expected {exp_skewness}"),
                );
            }
            if !close(*kurtosis, exp_kurtosis, atol, rtol) {
                return (
                    false,
                    format!("kurtosis mismatch: got {kurtosis}, expected {exp_kurtosis}"),
                );
            }
            (true, "describe_result match".to_string())
        }
        ("correlation_result", StatsObserved::Correlation { statistic, pvalue }) => {
            let exp_statistic = case.expected.statistic.unwrap_or(0.0);
            let exp_pvalue = case.expected.pvalue.unwrap_or(0.0);
            if !close(*statistic, exp_statistic, atol, rtol) {
                return (
                    false,
                    format!("statistic mismatch: got {statistic}, expected {exp_statistic}"),
                );
            }
            if !close(*pvalue, exp_pvalue, atol, rtol) {
                return (
                    false,
                    format!("pvalue mismatch: got {pvalue}, expected {exp_pvalue}"),
                );
            }
            (true, "correlation_result match".to_string())
        }
        (
            "linregress_result",
            StatsObserved::Linregress {
                slope,
                intercept,
                rvalue,
                pvalue,
                stderr,
            },
        ) => {
            let exp_slope = case.expected.slope.unwrap_or(0.0);
            let exp_intercept = case.expected.intercept.unwrap_or(0.0);
            let exp_rvalue = case.expected.rvalue.unwrap_or(0.0);
            let exp_pvalue = case.expected.pvalue.unwrap_or(0.0);
            let exp_stderr = case.expected.stderr.unwrap_or(0.0);
            if !close(*slope, exp_slope, atol, rtol) {
                return (
                    false,
                    format!("slope mismatch: got {slope}, expected {exp_slope}"),
                );
            }
            if !close(*intercept, exp_intercept, atol, rtol) {
                return (
                    false,
                    format!("intercept mismatch: got {intercept}, expected {exp_intercept}"),
                );
            }
            if !close(*rvalue, exp_rvalue, atol, rtol) {
                return (
                    false,
                    format!("rvalue mismatch: got {rvalue}, expected {exp_rvalue}"),
                );
            }
            if !close(*pvalue, exp_pvalue, atol, rtol) {
                return (
                    false,
                    format!("pvalue mismatch: got {pvalue}, expected {exp_pvalue}"),
                );
            }
            if !close(*stderr, exp_stderr, atol, rtol) {
                return (
                    false,
                    format!("stderr mismatch: got {stderr}, expected {exp_stderr}"),
                );
            }
            (true, "linregress_result match".to_string())
        }
        ("ttest_result", StatsObserved::Ttest { statistic, pvalue }) => {
            let exp_statistic = case.expected.statistic.unwrap_or(0.0);
            let exp_pvalue = case.expected.pvalue.unwrap_or(0.0);
            if !close(*statistic, exp_statistic, atol, rtol) {
                return (
                    false,
                    format!("statistic mismatch: got {statistic}, expected {exp_statistic}"),
                );
            }
            if !close(*pvalue, exp_pvalue, atol, rtol) {
                return (
                    false,
                    format!("pvalue mismatch: got {pvalue}, expected {exp_pvalue}"),
                );
            }
            (true, "ttest_result match".to_string())
        }
        ("goodness_result", StatsObserved::Goodness { statistic, pvalue }) => {
            let exp_statistic = case.expected.statistic.unwrap_or(0.0);
            let exp_pvalue = case.expected.pvalue.unwrap_or(0.0);
            if !close(*statistic, exp_statistic, atol, rtol) {
                return (
                    false,
                    format!("statistic mismatch: got {statistic}, expected {exp_statistic}"),
                );
            }
            if !close(*pvalue, exp_pvalue, atol, rtol) {
                return (
                    false,
                    format!("pvalue mismatch: got {pvalue}, expected {exp_pvalue}"),
                );
            }
            (true, "goodness_result match".to_string())
        }
        (kind, StatsObserved::Error(msg)) => (false, format!("error executing {kind}: {msg}")),
        (kind, observed) => (
            false,
            format!("kind/observed mismatch: expected {kind}, got {observed:?}"),
        ),
    }
}

pub fn run_stats_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: StatsPacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute_stats_case(case)
        })) {
            Ok(v) => v,
            Err(payload) => {
                case_results.push(CaseResult {
                    case_id: case.case_id().to_owned(),
                    passed: false,
                    message: format!(
                        "PANIC in execute_stats_case: {}",
                        panic_payload_message(payload)
                    ),
                });
                continue;
            }
        };
        let (passed, message) = compare_stats_outcome(case, &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// IntegrateCore harness (FSCI-P2C-013)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntegratePacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<IntegrateCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntegrateCase {
    pub case_id: String,
    pub category: String,
    pub mode: RuntimeMode,
    pub function: String,
    pub args: IntegrateArgs,
    pub expected: IntegrateExpected,
}

impl IntegrateCase {
    pub fn case_id(&self) -> &str {
        &self.case_id
    }
}

#[derive(Debug, Deserialize)]
pub struct IntegrateArgs {
    pub y: Option<Vec<f64>>,
    pub x: Option<Vec<f64>>,
    pub dx: Option<f64>,
    pub n: Option<usize>,
    pub func: Option<String>,
    pub a: Option<f64>,
    pub b: Option<f64>,
    pub lower: Option<Vec<f64>>,
    pub upper: Option<Vec<f64>>,
    #[serde(default, deserialize_with = "deserialize_maybe_nan_option_f64")]
    pub atol: Option<f64>,
    #[serde(default, deserialize_with = "deserialize_maybe_nan_option_f64")]
    pub rtol: Option<f64>,
    #[serde(default, deserialize_with = "deserialize_maybe_nan_option_f64")]
    pub first_step: Option<f64>,
    #[serde(default, deserialize_with = "deserialize_maybe_nan_option_f64")]
    pub max_step: Option<f64>,
    pub max_subdivisions: Option<usize>,
    pub points: Option<Vec<Vec<f64>>>,
    // br-9cla-2: IVP fields. Mirrors scipy.integrate.solve_ivp kwargs.
    pub rhs: Option<String>,
    pub t_span: Option<[f64; 2]>,
    pub y0: Option<Vec<f64>>,
    pub method: Option<String>,
    pub t_eval: Option<Vec<f64>>,
    pub event: Option<String>,
    /// br-r8ug: when present, request dense_output=true and sample
    /// the OdeSolution at these post-solve points. The runner emits
    /// the sampled (t, y) instead of the solver-step grid.
    pub dense_t_eval: Option<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
pub struct IntegrateExpected {
    pub kind: String,
    pub value: Option<serde_json::Value>,
    pub atol: Option<f64>,
    pub rtol: Option<f64>,
    #[serde(default)]
    pub contract_ref: String,
}

#[derive(Debug)]
enum IntegrateObserved {
    Scalar(f64),
    Array(Vec<f64>),
    /// IVP result (t, y) — `t` is the sample grid, `y[i]` is the i-th
    /// state-variable trajectory of length `t.len()`. br-9cla-2.
    IvpResult {
        t: Vec<f64>,
        y: Vec<Vec<f64>>,
    },
    Error(String),
}

fn execute_integrate_case(case: &IntegrateCase) -> IntegrateObserved {
    match case.function.as_str() {
        "trapezoid" => execute_integrate_trapezoid(case),
        "simpson" => execute_integrate_simpson(case),
        "cumulative_trapezoid" => execute_integrate_cumulative_trapezoid(case),
        "cumulative_simpson" => execute_integrate_cumulative_simpson(case),
        "romb" => execute_integrate_romb(case),
        "newton_cotes" => execute_integrate_newton_cotes(case),
        "fixed_quad" => execute_integrate_fixed_quad(case),
        "gauss_legendre" => execute_integrate_gauss_legendre(case),
        "cubature" => execute_integrate_cubature(case),
        "solve_ivp" => execute_integrate_solve_ivp(case),
        "odeint" => execute_integrate_odeint(case),
        "quad" => execute_integrate_quad(case),
        "quad_vec" => execute_integrate_quad_vec(case),
        "dblquad" => execute_integrate_dblquad(case),
        "tplquad" => execute_integrate_tplquad(case),
        "qmc_quad" => execute_integrate_qmc_quad(case),
        _ => IntegrateObserved::Error(format!("unknown function: {}", case.function)),
    }
}

/// br-9cla-2: named RHS dispatch for solve_ivp fixtures.
///
/// Keys MUST match `_build_ivp_rhs` in scipy_integrate_oracle.py so
/// both sides integrate the same ODE.
type IvpRhs = fn(f64, &[f64]) -> Vec<f64>;

fn ivp_rhs_by_name(name: &str) -> Option<IvpRhs> {
    match name {
        "exponential_decay" => Some(|_t, y| vec![-y[0]]),
        "stiff_decay" => Some(|_t, y| vec![-1000.0 * y[0]]),
        "linear_growth" => Some(|_t, _y| vec![1.0]),
        "harmonic_oscillator" => Some(|_t, y| vec![y[1], -y[0]]),
        // br-r8ug: classic stiff three-species Robertson chemistry.
        // y0' = -0.04*y0 + 1e4*y1*y2
        // y1' =  0.04*y0 - 1e4*y1*y2 - 3e7*y1^2
        // y2' =                       3e7*y1^2
        "robertson_chemistry" => Some(|_t, y| {
            let r1 = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
            let r3 = 3.0e7 * y[1] * y[1];
            let r2 = -r1 - r3;
            vec![r1, r2, r3]
        }),
        // br-r8ug: Van der Pol oscillator with mu=10 (mildly stiff).
        // y0' = y1
        // y1' = mu*(1 - y0^2)*y1 - y0
        "van_der_pol_mu10" => Some(|_t, y| {
            let mu = 10.0;
            vec![y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]]
        }),
        _ => None,
    }
}

fn event_y0_minus_half(_t: f64, y: &[f64]) -> f64 {
    y[0] - 0.5
}

fn ivp_event_by_name(name: &str) -> Option<fsci_integrate::EventFn> {
    match name {
        "y0_minus_half_terminal" => Some(event_y0_minus_half),
        _ => None,
    }
}

fn parse_solver_kind(method: &str) -> Option<fsci_integrate::SolverKind> {
    use fsci_integrate::SolverKind;
    match method {
        "RK45" => Some(SolverKind::Rk45),
        "RK23" => Some(SolverKind::Rk23),
        "DOP853" => Some(SolverKind::Dop853),
        "Radau" => Some(SolverKind::Radau),
        "BDF" => Some(SolverKind::Bdf),
        "LSODA" => Some(SolverKind::Lsoda),
        _ => None,
    }
}

fn execute_integrate_solve_ivp(case: &IntegrateCase) -> IntegrateObserved {
    let args = &case.args;
    let Some(rhs_name) = &args.rhs else {
        return IntegrateObserved::Error("missing rhs".to_string());
    };
    let Some(rhs_fn) = ivp_rhs_by_name(rhs_name) else {
        return IntegrateObserved::Error(format!("unknown rhs: {rhs_name}"));
    };
    let Some(t_span) = args.t_span else {
        return IntegrateObserved::Error("missing t_span".to_string());
    };
    let Some(y0) = args.y0.as_ref() else {
        return IntegrateObserved::Error("missing y0".to_string());
    };
    let method_str = args.method.as_deref().unwrap_or("RK45");
    let Some(method) = parse_solver_kind(method_str) else {
        return IntegrateObserved::Error(format!("unknown method: {method_str}"));
    };
    let rtol = args.rtol.unwrap_or(1e-3);
    let atol = args.atol.unwrap_or(1e-6);
    let events = match args.event.as_deref() {
        Some(event_name) => {
            let Some(event_fn) = ivp_event_by_name(event_name) else {
                return IntegrateObserved::Error(format!("unknown event: {event_name}"));
            };
            Some(vec![fsci_integrate::EventSpec::terminal(event_fn)])
        }
        None => None,
    };
    // br-r8ug: when dense_t_eval is supplied, request dense output and
    // sample the OdeSolution at those exact points. fsci's OdeSolution
    // is currently piecewise-constant on solver knots, so we interpolate
    // linearly here to approximate scipy's Hermite-spline dense output;
    // fixture tolerances should be loosened accordingly when comparing
    // against scipy.
    let dense_request = args.dense_t_eval.as_deref();
    let want_dense = dense_request.is_some();
    let effective_t_eval = dense_request.or(args.t_eval.as_deref());
    let opts = fsci_integrate::SolveIvpOptions {
        t_span: (t_span[0], t_span[1]),
        y0,
        method,
        t_eval: effective_t_eval,
        dense_output: want_dense,
        events,
        rtol,
        atol: fsci_integrate::ToleranceValue::Scalar(atol),
        first_step: args.first_step,
        max_step: args.max_step.unwrap_or(f64::INFINITY),
        mode: case.mode,
    };
    let mut rhs_mut = rhs_fn;
    match fsci_integrate::solve_ivp(&mut rhs_mut, &opts) {
        Ok(res) => {
            let n_vars = res.y.first().map_or(0, Vec::len);
            let mut y_t: Vec<Vec<f64>> = (0..n_vars)
                .map(|_| Vec::with_capacity(res.y.len()))
                .collect();
            for step in &res.y {
                for (i, &v) in step.iter().enumerate() {
                    y_t[i].push(v);
                }
            }
            IntegrateObserved::IvpResult { t: res.t, y: y_t }
        }
        Err(e) => IntegrateObserved::Error(format!("{e:?}")),
    }
}

fn execute_integrate_odeint(case: &IntegrateCase) -> IntegrateObserved {
    let args = &case.args;
    let Some(rhs_name) = &args.rhs else {
        return IntegrateObserved::Error("missing rhs".to_string());
    };
    let Some(rhs_fn) = ivp_rhs_by_name(rhs_name) else {
        return IntegrateObserved::Error(format!("unknown rhs: {rhs_name}"));
    };
    let Some(y0) = args.y0.as_ref() else {
        return IntegrateObserved::Error("missing y0".to_string());
    };
    let Some(t_eval) = args.t_eval.as_ref() else {
        return IntegrateObserved::Error("missing t_eval".to_string());
    };

    let mut rhs_mut = |y: &[f64], t: f64| rhs_fn(t, y);
    match fsci_integrate::odeint(&mut rhs_mut, y0, t_eval) {
        Ok(y_steps) => {
            let n_vars = y_steps.first().map_or(0, Vec::len);
            let mut y_t: Vec<Vec<f64>> = (0..n_vars)
                .map(|_| Vec::with_capacity(y_steps.len()))
                .collect();
            for step in &y_steps {
                for (i, &v) in step.iter().enumerate() {
                    y_t[i].push(v);
                }
            }
            IntegrateObserved::IvpResult {
                t: t_eval.clone(),
                y: y_t,
            }
        }
        Err(e) => IntegrateObserved::Error(format!("{e:?}")),
    }
}

fn execute_integrate_trapezoid(case: &IntegrateCase) -> IntegrateObserved {
    let y = match &case.args.y {
        Some(v) => v.clone(),
        None => return IntegrateObserved::Error("missing y".to_string()),
    };
    let x = match &case.args.x {
        Some(v) => v.clone(),
        None => return IntegrateObserved::Error("missing x".to_string()),
    };
    match trapezoid(&y, &x) {
        Ok(result) => IntegrateObserved::Scalar(result.integral),
        Err(e) => IntegrateObserved::Error(format!("{e}")),
    }
}

fn execute_integrate_simpson(case: &IntegrateCase) -> IntegrateObserved {
    let y = match &case.args.y {
        Some(v) => v.clone(),
        None => return IntegrateObserved::Error("missing y".to_string()),
    };
    let x = match &case.args.x {
        Some(v) => v.clone(),
        None => return IntegrateObserved::Error("missing x".to_string()),
    };
    match simpson(&y, &x) {
        Ok(result) => IntegrateObserved::Scalar(result.integral),
        Err(e) => IntegrateObserved::Error(format!("{e}")),
    }
}

fn execute_integrate_cumulative_trapezoid(case: &IntegrateCase) -> IntegrateObserved {
    let y = match &case.args.y {
        Some(v) => v.clone(),
        None => return IntegrateObserved::Error("missing y".to_string()),
    };
    let x = match &case.args.x {
        Some(v) => v.clone(),
        None => return IntegrateObserved::Error("missing x".to_string()),
    };
    match cumulative_trapezoid(&y, &x) {
        Ok(result) => IntegrateObserved::Array(result),
        Err(e) => IntegrateObserved::Error(format!("{e}")),
    }
}

fn execute_integrate_cumulative_simpson(case: &IntegrateCase) -> IntegrateObserved {
    let y = match &case.args.y {
        Some(v) => v.clone(),
        None => return IntegrateObserved::Error("missing y".to_string()),
    };
    let x = match &case.args.x {
        Some(v) => v.clone(),
        None => return IntegrateObserved::Error("missing x".to_string()),
    };
    match cumulative_simpson(&y, &x) {
        Ok(result) => IntegrateObserved::Array(result),
        Err(e) => IntegrateObserved::Error(format!("{e}")),
    }
}

fn execute_integrate_romb(case: &IntegrateCase) -> IntegrateObserved {
    let y = match &case.args.y {
        Some(v) => v.clone(),
        None => return IntegrateObserved::Error("missing y".to_string()),
    };
    let dx = case.args.dx.unwrap_or(1.0);
    match romb(&y, dx) {
        Ok(result) => IntegrateObserved::Scalar(result),
        Err(e) => IntegrateObserved::Error(format!("{e}")),
    }
}

fn execute_integrate_newton_cotes(case: &IntegrateCase) -> IntegrateObserved {
    let n = match case.args.n {
        Some(v) => v,
        None => return IntegrateObserved::Error("missing n".to_string()),
    };
    match newton_cotes(n) {
        Ok(weights) => IntegrateObserved::Array(weights),
        Err(e) => IntegrateObserved::Error(format!("{e}")),
    }
}

fn make_integrate_func(func_str: &str) -> Option<Box<dyn Fn(f64) -> f64>> {
    match func_str {
        "x" => Some(Box::new(|x| x)),
        "x^2" => Some(Box::new(|x| x * x)),
        "x^3" => Some(Box::new(|x| x * x * x)),
        "sin(x)" => Some(Box::new(|x| x.sin())),
        "cos(x)" => Some(Box::new(|x| x.cos())),
        "exp(x)" => Some(Box::new(|x| x.exp())),
        _ => None,
    }
}

fn execute_integrate_fixed_quad(case: &IntegrateCase) -> IntegrateObserved {
    let func_str = match &case.args.func {
        Some(s) => s.as_str(),
        None => return IntegrateObserved::Error("missing func".to_string()),
    };
    let a = match case.args.a {
        Some(v) => v,
        None => return IntegrateObserved::Error("missing a".to_string()),
    };
    let b = match case.args.b {
        Some(v) => v,
        None => return IntegrateObserved::Error("missing b".to_string()),
    };
    let n = match case.args.n {
        Some(v) => v,
        None => return IntegrateObserved::Error("missing n".to_string()),
    };
    let f = match make_integrate_func(func_str) {
        Some(f) => f,
        None => return IntegrateObserved::Error(format!("unknown func: {func_str}")),
    };
    match fixed_quad(&*f, a, b, n) {
        Ok((integral, _)) => IntegrateObserved::Scalar(integral),
        Err(e) => IntegrateObserved::Error(format!("{e}")),
    }
}

fn execute_integrate_gauss_legendre(case: &IntegrateCase) -> IntegrateObserved {
    let func_str = match &case.args.func {
        Some(s) => s.as_str(),
        None => return IntegrateObserved::Error("missing func".to_string()),
    };
    let a = match case.args.a {
        Some(v) => v,
        None => return IntegrateObserved::Error("missing a".to_string()),
    };
    let b = match case.args.b {
        Some(v) => v,
        None => return IntegrateObserved::Error("missing b".to_string()),
    };
    let n = match case.args.n {
        Some(v) => v,
        None => return IntegrateObserved::Error("missing n".to_string()),
    };
    let f = match make_integrate_func(func_str) {
        Some(f) => f,
        None => return IntegrateObserved::Error(format!("unknown func: {func_str}")),
    };
    let result = gauss_legendre(&*f, a, b, n);
    IntegrateObserved::Scalar(result)
}

type CubatureScalarFn = Box<dyn Fn(&[f64]) -> f64>;
type CubatureVectorFn = Box<dyn Fn(&[f64]) -> Vec<f64>>;

fn make_cubature_scalar_func(func_str: &str) -> Option<CubatureScalarFn> {
    match func_str {
        "constant_1" => Some(Box::new(|_| 1.0)),
        "x0*x1" => Some(Box::new(|x| x[0] * x[1])),
        "x0+x1" => Some(Box::new(|x| x[0] + x[1])),
        _ => None,
    }
}

fn make_cubature_vector_func(func_str: &str) -> Option<CubatureVectorFn> {
    match func_str {
        "powers_1d" => Some(Box::new(|x| vec![x[0], x[0] * x[0]])),
        "zero_dim_pair" => Some(Box::new(|_| vec![42.0, -7.0])),
        _ => None,
    }
}

// br-9cla-5: scalar integrand registry for quad().
//
// Keys MUST match scipy_integrate_oracle.py's _build_callable so both
// sides integrate the same function.
fn make_integrate_quad_func(name: &str) -> Option<Box<dyn Fn(f64) -> f64>> {
    make_integrate_func(name)
}

// br-9cla-5: vector integrand registry for quad_vec().
//
// Returns a fixed-length Vec<f64>. The dimensions are encoded into the
// name ("linear_square" always returns 2 components).
fn make_integrate_quad_vec_func(name: &str) -> Option<Box<dyn Fn(f64) -> Vec<f64>>> {
    match name {
        "linear_square" => Some(Box::new(|x| vec![x, x * x])),
        _ => None,
    }
}

// br-9cla-5: 2D integrand registry for dblquad().
//
// scipy.integrate.dblquad takes f(y, x) (inner variable first); we
// follow the same convention since fsci_integrate::dblquad does too.
// Inner bounds are named-integrand specific and returned alongside
// the callable so the oracle side stays in sync.
#[allow(clippy::type_complexity)]
fn make_integrate_dblquad_func(name: &str) -> Option<(Box<dyn Fn(f64, f64) -> f64>, f64, f64)> {
    match name {
        // ∫_0^b ∫_y_lo^y_hi x*y dy dx (outer bounds come from the case).
        "xy_prod_unit_y" => Some((Box::new(|y, x| x * y), 0.0, 1.0)),
        _ => None,
    }
}

// br-9cla-5: 3D integrand registry for tplquad().
//
// scipy.integrate.tplquad takes f(z, y, x) with z innermost. fsci
// mirrors this. Inner+middle bounds are baked into the named integrand.
#[allow(clippy::type_complexity)]
fn make_integrate_tplquad_func(
    name: &str,
) -> Option<(Box<dyn Fn(f64, f64, f64) -> f64>, f64, f64, f64, f64)> {
    match name {
        // ∫_0^b ∫_0^1 ∫_0^1 x*y*z dz dy dx (outer bounds from the case).
        "xyz_prod_unit_yz" => Some((Box::new(|z, y, x| x * y * z), 0.0, 1.0, 0.0, 1.0)),
        _ => None,
    }
}

// br-gm7n: vector-input integrand registry for qmc_quad. Keys are
// shared with fixture cases — point[j] is the j-th coordinate.
#[allow(clippy::type_complexity)]
fn make_integrate_qmc_func(name: &str) -> Option<Box<dyn Fn(&[f64]) -> f64>> {
    match name {
        "x_squared_1d" => Some(Box::new(|p| p[0] * p[0])),
        "sin_x_1d" => Some(Box::new(|p| p[0].sin())),
        "x_plus_y_2d" => Some(Box::new(|p| p[0] + p[1])),
        "exp_neg_xy_2d" => Some(Box::new(|p| (-p[0] * p[1]).exp())),
        _ => None,
    }
}

fn execute_integrate_quad(case: &IntegrateCase) -> IntegrateObserved {
    let Some(func_str) = case.args.func.as_deref() else {
        return IntegrateObserved::Error("quad: missing func".to_string());
    };
    let Some(a) = case.args.a else {
        return IntegrateObserved::Error("quad: missing a".to_string());
    };
    let Some(b) = case.args.b else {
        return IntegrateObserved::Error("quad: missing b".to_string());
    };
    let Some(f) = make_integrate_quad_func(func_str) else {
        return IntegrateObserved::Error(format!("quad: unknown func: {func_str}"));
    };
    let options = QuadOptions {
        epsabs: case.args.atol.unwrap_or(1.49e-8),
        epsrel: case.args.rtol.unwrap_or(1.49e-8),
        limit: case.args.max_subdivisions.unwrap_or(50),
    };
    match quad(&*f, a, b, options) {
        Ok(result) => IntegrateObserved::Scalar(result.integral),
        Err(e) => IntegrateObserved::Error(format!("{e}")),
    }
}

fn execute_integrate_quad_vec(case: &IntegrateCase) -> IntegrateObserved {
    let Some(func_str) = case.args.func.as_deref() else {
        return IntegrateObserved::Error("quad_vec: missing func".to_string());
    };
    let Some(a) = case.args.a else {
        return IntegrateObserved::Error("quad_vec: missing a".to_string());
    };
    let Some(b) = case.args.b else {
        return IntegrateObserved::Error("quad_vec: missing b".to_string());
    };
    let Some(f) = make_integrate_quad_vec_func(func_str) else {
        return IntegrateObserved::Error(format!("quad_vec: unknown func: {func_str}"));
    };
    let options = QuadOptions {
        epsabs: case.args.atol.unwrap_or(1.49e-8),
        epsrel: case.args.rtol.unwrap_or(1.49e-8),
        limit: case.args.max_subdivisions.unwrap_or(50),
    };
    match quad_vec(&*f, a, b, options) {
        Ok(result) => IntegrateObserved::Array(result.integral),
        Err(e) => IntegrateObserved::Error(format!("{e}")),
    }
}

fn execute_integrate_dblquad(case: &IntegrateCase) -> IntegrateObserved {
    let Some(func_str) = case.args.func.as_deref() else {
        return IntegrateObserved::Error("dblquad: missing func".to_string());
    };
    let Some(a) = case.args.a else {
        return IntegrateObserved::Error("dblquad: missing a".to_string());
    };
    let Some(b) = case.args.b else {
        return IntegrateObserved::Error("dblquad: missing b".to_string());
    };
    let Some((f, y_lo, y_hi)) = make_integrate_dblquad_func(func_str) else {
        return IntegrateObserved::Error(format!("dblquad: unknown func: {func_str}"));
    };
    let options = DblquadOptions {
        epsabs: case.args.atol.unwrap_or(1.49e-8),
        epsrel: case.args.rtol.unwrap_or(1.49e-8),
        limit: case.args.max_subdivisions.unwrap_or(50),
    };
    match dblquad(&*f, a, b, |_| y_lo, |_| y_hi, options) {
        Ok(result) => IntegrateObserved::Scalar(result.integral),
        Err(e) => IntegrateObserved::Error(format!("{e}")),
    }
}

fn execute_integrate_tplquad(case: &IntegrateCase) -> IntegrateObserved {
    let Some(func_str) = case.args.func.as_deref() else {
        return IntegrateObserved::Error("tplquad: missing func".to_string());
    };
    let Some(a) = case.args.a else {
        return IntegrateObserved::Error("tplquad: missing a".to_string());
    };
    let Some(b) = case.args.b else {
        return IntegrateObserved::Error("tplquad: missing b".to_string());
    };
    let Some((f, y_lo, y_hi, z_lo, z_hi)) = make_integrate_tplquad_func(func_str) else {
        return IntegrateObserved::Error(format!("tplquad: unknown func: {func_str}"));
    };
    let options = DblquadOptions {
        epsabs: case.args.atol.unwrap_or(1.49e-8),
        epsrel: case.args.rtol.unwrap_or(1.49e-8),
        limit: case.args.max_subdivisions.unwrap_or(50),
    };
    match tplquad(
        &*f,
        a,
        b,
        |_| y_lo,
        |_| y_hi,
        |_, _| z_lo,
        |_, _| z_hi,
        options,
    ) {
        Ok(result) => IntegrateObserved::Scalar(result.integral),
        Err(e) => IntegrateObserved::Error(format!("{e}")),
    }
}

// br-gm7n: qmc_quad executor. Reuses IntegrateArgs.lower/upper for the
// hyper-rectangle, args.func for the named integrand, args.n for
// n_points (default 1024), and a fixed n_estimates=8 to mirror scipy's
// default. Halton is deterministic and unscrambled, so estimates are
// disjoint segments of the same sequence rather than independent draws.
fn execute_integrate_qmc_quad(case: &IntegrateCase) -> IntegrateObserved {
    let Some(func_str) = case.args.func.as_deref() else {
        return IntegrateObserved::Error("qmc_quad: missing func".to_string());
    };
    let Some(lb) = case.args.lower.as_ref() else {
        return IntegrateObserved::Error("qmc_quad: missing lower".to_string());
    };
    let Some(ub) = case.args.upper.as_ref() else {
        return IntegrateObserved::Error("qmc_quad: missing upper".to_string());
    };
    let Some(f) = make_integrate_qmc_func(func_str) else {
        return IntegrateObserved::Error(format!("qmc_quad: unknown func: {func_str}"));
    };
    let n_points = case.args.n.unwrap_or(1024);
    let n_estimates = 8usize;
    match qmc_quad(&*f, lb, ub, n_estimates, n_points) {
        Ok(result) => IntegrateObserved::Scalar(result.integral),
        Err(e) => IntegrateObserved::Error(format!("{e}")),
    }
}

fn cubature_options_from_case(case: &IntegrateCase) -> CubatureOptions {
    let mut options = CubatureOptions::default();
    if let Some(atol) = case.args.atol {
        options.atol = atol;
    }
    if let Some(rtol) = case.args.rtol {
        options.rtol = rtol;
    }
    if let Some(max_subdivisions) = case.args.max_subdivisions {
        options.max_subdivisions = max_subdivisions;
    }
    if let Some(points) = &case.args.points {
        options.points = points.clone();
    }
    options
}

fn execute_integrate_cubature(case: &IntegrateCase) -> IntegrateObserved {
    let func_str = match &case.args.func {
        Some(s) => s.as_str(),
        None => return IntegrateObserved::Error("missing func".to_string()),
    };
    let lower = match &case.args.lower {
        Some(v) => v.clone(),
        None => return IntegrateObserved::Error("missing lower".to_string()),
    };
    let upper = match &case.args.upper {
        Some(v) => v.clone(),
        None => return IntegrateObserved::Error("missing upper".to_string()),
    };
    let options = cubature_options_from_case(case);

    match case.expected.kind.as_str() {
        "scalar" => {
            let Some(f) = make_cubature_scalar_func(func_str) else {
                return IntegrateObserved::Error(format!("unknown cubature func: {func_str}"));
            };
            match cubature_scalar(&*f, &lower, &upper, options) {
                Ok(result) => IntegrateObserved::Scalar(result.estimate),
                Err(e) => IntegrateObserved::Error(format!("{e}")),
            }
        }
        "array" => {
            let Some(f) = make_cubature_vector_func(func_str) else {
                return IntegrateObserved::Error(format!("unknown cubature func: {func_str}"));
            };
            match cubature(&*f, &lower, &upper, options) {
                Ok(result) => IntegrateObserved::Array(result.estimate),
                Err(e) => IntegrateObserved::Error(format!("{e}")),
            }
        }
        "error" => {
            if let Some(f) = make_cubature_scalar_func(func_str) {
                match cubature_scalar(&*f, &lower, &upper, options) {
                    Ok(result) => IntegrateObserved::Scalar(result.estimate),
                    Err(e) => IntegrateObserved::Error(format!("{e}")),
                }
            } else if let Some(f) = make_cubature_vector_func(func_str) {
                match cubature(&*f, &lower, &upper, options) {
                    Ok(result) => IntegrateObserved::Array(result.estimate),
                    Err(e) => IntegrateObserved::Error(format!("{e}")),
                }
            } else {
                IntegrateObserved::Error(format!("unknown cubature func: {func_str}"))
            }
        }
        _ => IntegrateObserved::Error(format!(
            "unknown expected integrate kind: {}",
            case.expected.kind
        )),
    }
}

fn compare_integrate_outcome(case: &IntegrateCase, observed: &IntegrateObserved) -> (bool, String) {
    let atol = case.expected.atol.unwrap_or(1e-12);
    let rtol = case.expected.rtol.unwrap_or(1e-12);

    fn close(a: f64, b: f64, atol: f64, rtol: f64) -> bool {
        allclose_scalar(a, b, atol, rtol)
    }

    match (case.expected.kind.as_str(), observed) {
        ("scalar", IntegrateObserved::Scalar(got)) => {
            let expected = match &case.expected.value {
                Some(serde_json::Value::Number(n)) => n.as_f64().unwrap_or(0.0),
                _ => return (false, "expected scalar value missing".to_string()),
            };
            if close(*got, expected, atol, rtol) {
                (true, format!("scalar match: {got}"))
            } else {
                (
                    false,
                    format!("scalar mismatch: got {got}, expected {expected}"),
                )
            }
        }
        ("array", IntegrateObserved::Array(got)) => {
            let expected: Vec<f64> = match &case.expected.value {
                Some(serde_json::Value::Array(arr)) => {
                    arr.iter().filter_map(|v| v.as_f64()).collect()
                }
                _ => return (false, "expected array value missing".to_string()),
            };
            if got.len() != expected.len() {
                return (
                    false,
                    format!(
                        "array length mismatch: got {}, expected {}",
                        got.len(),
                        expected.len()
                    ),
                );
            }
            for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
                if !close(g, e, atol, rtol) {
                    return (
                        false,
                        format!("array mismatch at index {i}: got {g}, expected {e}"),
                    );
                }
            }
            (true, format!("array match ({} elements)", got.len()))
        }
        ("ivp_result", IntegrateObserved::IvpResult { t, y }) => {
            // br-9cla-2: expected value is a JSON object {t: [...], y: [[...], ...]}
            let expected_obj = match &case.expected.value {
                Some(serde_json::Value::Object(o)) => o,
                _ => return (false, "expected ivp_result object missing".to_string()),
            };
            let Some(serde_json::Value::Array(exp_t)) = expected_obj.get("t") else {
                return (false, "expected.value.t missing or not array".to_string());
            };
            let Some(serde_json::Value::Array(exp_y)) = expected_obj.get("y") else {
                return (false, "expected.value.y missing or not array".to_string());
            };
            if exp_t.len() != t.len() {
                return (
                    false,
                    format!(
                        "ivp t length mismatch: got {}, expected {}",
                        t.len(),
                        exp_t.len()
                    ),
                );
            }
            for (i, (g, e)) in t.iter().zip(exp_t.iter()).enumerate() {
                let ev = e.as_f64().unwrap_or(f64::NAN);
                if !close(*g, ev, atol, rtol) {
                    return (
                        false,
                        format!("ivp t[{i}] mismatch: got {g}, expected {ev}"),
                    );
                }
            }
            if exp_y.len() != y.len() {
                return (
                    false,
                    format!(
                        "ivp y outer length mismatch: got {}, expected {}",
                        y.len(),
                        exp_y.len()
                    ),
                );
            }
            for (var_idx, (got_row, exp_row_v)) in y.iter().zip(exp_y.iter()).enumerate() {
                let exp_row = match exp_row_v {
                    serde_json::Value::Array(a) => a,
                    _ => return (false, format!("ivp y[{var_idx}] not an array")),
                };
                if got_row.len() != exp_row.len() {
                    return (
                        false,
                        format!(
                            "ivp y[{var_idx}] length mismatch: got {}, expected {}",
                            got_row.len(),
                            exp_row.len()
                        ),
                    );
                }
                for (j, (gv, ev_json)) in got_row.iter().zip(exp_row.iter()).enumerate() {
                    let ev = ev_json.as_f64().unwrap_or(f64::NAN);
                    if !close(*gv, ev, atol, rtol) {
                        return (
                            false,
                            format!("ivp y[{var_idx}][{j}] mismatch: got {gv}, expected {ev}"),
                        );
                    }
                }
            }
            (
                true,
                format!("ivp match ({} steps, {} vars)", t.len(), y.len()),
            )
        }
        ("error", IntegrateObserved::Error(msg))
        | ("error_kind", IntegrateObserved::Error(msg)) => {
            let expected = match &case.expected.value {
                Some(serde_json::Value::String(value)) => value,
                _ => return (false, "expected error value missing".to_string()),
            };
            if msg == expected {
                (true, format!("error match: {msg}"))
            } else {
                (
                    false,
                    format!("error mismatch: got {msg}, expected {expected}"),
                )
            }
        }
        (_, IntegrateObserved::Error(msg)) => (false, format!("error: {msg}")),
        (kind, observed) => (
            false,
            format!("kind/observed mismatch: expected {kind}, got {observed:?}"),
        ),
    }
}

fn compare_integrate_case_differential(
    case: &IntegrateCase,
    observed: &IntegrateObserved,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    let tolerance = resolve_integrate_contract_tolerance(
        Some(case.expected.contract_ref.as_str()),
        case.expected.atol,
        case.expected.rtol,
    );

    match (case.expected.kind.as_str(), observed) {
        ("scalar", IntegrateObserved::Scalar(actual)) => {
            let expected = match &case.expected.value {
                Some(serde_json::Value::Number(value)) => match value.as_f64() {
                    Some(value) => value,
                    None => {
                        return (
                            false,
                            "expected scalar value must be finite".to_owned(),
                            None,
                            None,
                        );
                    }
                },
                _ => {
                    return (
                        false,
                        "expected scalar value missing".to_owned(),
                        None,
                        None,
                    );
                }
            };
            let diff = (*actual - expected).abs();
            let pass = allclose_scalar(*actual, expected, tolerance.atol, tolerance.rtol);
            let msg = if pass {
                format!("integrate scalar matched (diff={diff:.2e})")
            } else {
                format!(
                    "integrate scalar mismatch: expected={expected:.16e}, got={actual:.16e}, diff={diff:.2e}, atol={:.2e}, rtol={:.2e}",
                    tolerance.atol, tolerance.rtol
                )
            };
            (pass, msg, Some(diff), Some(tolerance))
        }
        ("array", IntegrateObserved::Array(actual)) => {
            let expected = match &case.expected.value {
                Some(serde_json::Value::Array(values)) => values
                    .iter()
                    .map(serde_json::Value::as_f64)
                    .collect::<Option<Vec<_>>>(),
                _ => None,
            };
            let expected = match expected {
                Some(expected) => expected,
                None => {
                    return (false, "expected array value missing".to_owned(), None, None);
                }
            };

            let diff = max_diff_vec(actual, &expected);
            let pass = allclose_vec(actual, &expected, tolerance.atol, tolerance.rtol);
            let msg = if pass {
                format!(
                    "integrate array matched len={} (max_diff={diff:.2e})",
                    actual.len()
                )
            } else if actual.len() != expected.len() {
                format!(
                    "integrate array length mismatch: expected={}, got={}",
                    expected.len(),
                    actual.len()
                )
            } else {
                format!(
                    "integrate array mismatch: max_diff={diff:.2e}, atol={:.2e}, rtol={:.2e}",
                    tolerance.atol, tolerance.rtol
                )
            };
            (pass, msg, Some(diff), Some(tolerance))
        }
        ("error", IntegrateObserved::Error(actual))
        | ("error_kind", IntegrateObserved::Error(actual)) => {
            let expected = match &case.expected.value {
                Some(serde_json::Value::String(value)) => value,
                _ => {
                    return (false, "expected error value missing".to_owned(), None, None);
                }
            };
            let pass = actual == expected;
            let msg = if pass {
                format!("integrate error matched ({actual})")
            } else {
                format!("integrate error mismatch: expected={expected}, got={actual}")
            };
            (pass, msg, None, None)
        }
        ("ivp_result", IntegrateObserved::IvpResult { t, y }) => {
            // br-9cla-2/9cla-3: element-wise compare under case tolerances.
            let expected_obj = match &case.expected.value {
                Some(serde_json::Value::Object(o)) => o,
                _ => {
                    return (
                        false,
                        "expected ivp_result object missing".to_owned(),
                        None,
                        None,
                    );
                }
            };
            let exp_t = match expected_obj.get("t") {
                Some(serde_json::Value::Array(a)) => a,
                _ => {
                    return (
                        false,
                        "expected.value.t missing/not array".to_owned(),
                        None,
                        None,
                    );
                }
            };
            let exp_y = match expected_obj.get("y") {
                Some(serde_json::Value::Array(a)) => a,
                _ => {
                    return (
                        false,
                        "expected.value.y missing/not array".to_owned(),
                        None,
                        None,
                    );
                }
            };
            let mut max_diff = 0.0_f64;
            if exp_t.len() != t.len() {
                return (
                    false,
                    format!(
                        "ivp t length mismatch: expected={}, got={}",
                        exp_t.len(),
                        t.len()
                    ),
                    None,
                    Some(tolerance),
                );
            }
            for (got_v, exp_v) in t.iter().zip(exp_t.iter()) {
                let ev = exp_v.as_f64().unwrap_or(f64::NAN);
                max_diff = max_diff.max((got_v - ev).abs());
                if !allclose_scalar(*got_v, ev, tolerance.atol, tolerance.rtol) {
                    return (
                        false,
                        format!(
                            "ivp t mismatch: got {got_v}, expected {ev}, diff={:.2e}",
                            (got_v - ev).abs()
                        ),
                        Some(max_diff),
                        Some(tolerance),
                    );
                }
            }
            if exp_y.len() != y.len() {
                return (
                    false,
                    format!(
                        "ivp y outer length mismatch: expected={}, got={}",
                        exp_y.len(),
                        y.len()
                    ),
                    Some(max_diff),
                    Some(tolerance),
                );
            }
            for (var_idx, (got_row, exp_row_v)) in y.iter().zip(exp_y.iter()).enumerate() {
                let exp_row = match exp_row_v {
                    serde_json::Value::Array(a) => a,
                    _ => {
                        return (
                            false,
                            format!("ivp y[{var_idx}] not an array"),
                            Some(max_diff),
                            Some(tolerance),
                        );
                    }
                };
                if got_row.len() != exp_row.len() {
                    return (
                        false,
                        format!(
                            "ivp y[{var_idx}] length mismatch: expected={}, got={}",
                            exp_row.len(),
                            got_row.len()
                        ),
                        Some(max_diff),
                        Some(tolerance),
                    );
                }
                for (j, (gv, ev_json)) in got_row.iter().zip(exp_row.iter()).enumerate() {
                    let ev = ev_json.as_f64().unwrap_or(f64::NAN);
                    max_diff = max_diff.max((gv - ev).abs());
                    if !allclose_scalar(*gv, ev, tolerance.atol, tolerance.rtol) {
                        return (
                            false,
                            format!(
                                "ivp y[{var_idx}][{j}] mismatch: got {gv}, expected {ev}, diff={:.2e}",
                                (gv - ev).abs()
                            ),
                            Some(max_diff),
                            Some(tolerance),
                        );
                    }
                }
            }
            (
                true,
                format!(
                    "integrate ivp matched ({} steps, {} vars, max_diff={max_diff:.2e})",
                    t.len(),
                    y.len()
                ),
                Some(max_diff),
                Some(tolerance),
            )
        }
        (expected_kind, actual) => (
            false,
            format!("shape mismatch: expected kind={expected_kind}, got {actual:?}"),
            None,
            None,
        ),
    }
}

fn compare_stats_case_differential(
    case: &StatsCase,
    observed: &StatsObserved,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    let (passed, message) = compare_stats_outcome(case, observed);
    let tolerance = || {
        resolve_stats_contract_tolerance(
            Some(case.expected.contract_ref.as_str()),
            case.expected.atol,
            case.expected.rtol,
        )
    };

    match (case.expected.kind.as_str(), observed) {
        ("scalar", StatsObserved::Scalar(actual)) => (
            passed,
            message,
            case.expected
                .value
                .map(|expected| (*actual - expected).abs()),
            Some(tolerance()),
        ),
        ("array", StatsObserved::Array(actual)) => {
            let expected = case.expected.array_value.as_deref().unwrap_or(&[]);
            (
                passed,
                message,
                Some(max_diff_vec(actual, expected)),
                Some(tolerance()),
            )
        }
        (
            "describe_result",
            StatsObserved::Describe {
                minmax,
                mean,
                variance,
                skewness,
                kurtosis,
                ..
            },
        ) => {
            let expected = [
                (minmax[0] - case.expected.minmax.unwrap_or([0.0, 0.0])[0]).abs(),
                (minmax[1] - case.expected.minmax.unwrap_or([0.0, 0.0])[1]).abs(),
                (*mean - case.expected.mean.unwrap_or(0.0)).abs(),
                (*variance - case.expected.variance.unwrap_or(0.0)).abs(),
                (*skewness - case.expected.skewness.unwrap_or(0.0)).abs(),
                (*kurtosis - case.expected.kurtosis.unwrap_or(0.0)).abs(),
            ];
            let diff = expected.into_iter().fold(0.0_f64, f64::max);
            (passed, message, Some(diff), Some(tolerance()))
        }
        ("correlation_result", StatsObserved::Correlation { statistic, pvalue })
        | ("ttest_result", StatsObserved::Ttest { statistic, pvalue })
        | ("goodness_result", StatsObserved::Goodness { statistic, pvalue }) => {
            let diff = (*statistic - case.expected.statistic.unwrap_or(0.0))
                .abs()
                .max((*pvalue - case.expected.pvalue.unwrap_or(0.0)).abs());
            (passed, message, Some(diff), Some(tolerance()))
        }
        (
            "linregress_result",
            StatsObserved::Linregress {
                slope,
                intercept,
                rvalue,
                pvalue,
                stderr,
            },
        ) => {
            let diff = [
                (*slope - case.expected.slope.unwrap_or(0.0)).abs(),
                (*intercept - case.expected.intercept.unwrap_or(0.0)).abs(),
                (*rvalue - case.expected.rvalue.unwrap_or(0.0)).abs(),
                (*pvalue - case.expected.pvalue.unwrap_or(0.0)).abs(),
                (*stderr - case.expected.stderr.unwrap_or(0.0)).abs(),
            ]
            .into_iter()
            .fold(0.0_f64, f64::max);
            (passed, message, Some(diff), Some(tolerance()))
        }
        _ => (passed, message, None, None),
    }
}

fn compare_signal_case_differential(
    case: &SignalCase,
    observed: &SignalObserved,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    let (passed, message) = compare_signal_outcome(case, observed);
    let tolerance = || {
        resolve_signal_contract_tolerance(
            Some(case.expected.contract_ref.as_str()),
            case.expected.atol,
            case.expected.rtol,
        )
    };

    match (case.expected.kind.as_str(), observed) {
        ("array", SignalObserved::Array(actual)) => {
            let expected = case.expected.value.as_deref().unwrap_or(&[]);
            (
                passed,
                message,
                Some(max_diff_vec(actual, expected)),
                Some(tolerance()),
            )
        }
        ("indices", SignalObserved::Indices(_)) => (passed, message, None, None),
        ("iir_coeffs", SignalObserved::IirCoeffs { b, a }) => {
            let exp_b = case.expected.b.as_deref().unwrap_or(&[]);
            let exp_a = case.expected.a.as_deref().unwrap_or(&[]);
            let diff = max_diff_vec(b, exp_b).max(max_diff_vec(a, exp_a));
            (passed, message, Some(diff), Some(tolerance()))
        }
        ("freqz_result", SignalObserved::Freqz { w, h_mag, h_phase }) => {
            let exp_w = case.expected.w.as_deref().unwrap_or(&[]);
            let exp_h_mag = case.expected.h_mag.as_deref().unwrap_or(&[]);
            let exp_h_phase = case.expected.h_phase.as_deref().unwrap_or(&[]);
            let diff = max_diff_vec(w, exp_w)
                .max(max_diff_vec(h_mag, exp_h_mag))
                .max(max_diff_vec(h_phase, exp_h_phase));
            (passed, message, Some(diff), Some(tolerance()))
        }
        ("complex_array", SignalObserved::ComplexArray { real, imag }) => {
            let exp_real = case.expected.real.as_deref().unwrap_or(&[]);
            let exp_imag = case.expected.imag.as_deref().unwrap_or(&[]);
            let diff = max_diff_vec(real, exp_real).max(max_diff_vec(imag, exp_imag));
            (passed, message, Some(diff), Some(tolerance()))
        }
        _ => (passed, message, None, None),
    }
}

fn compare_spatial_case_differential(
    case: &SpatialCase,
    observed: &SpatialObserved,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    let (passed, message) = compare_spatial_outcome(case, observed);
    let tolerance = || {
        resolve_spatial_contract_tolerance(
            Some(case.expected.contract_ref.as_str()),
            case.expected.atol,
            case.expected.rtol,
        )
    };

    match (case.expected.kind.as_str(), observed) {
        ("scalar", SpatialObserved::Scalar(actual)) => {
            let expected = case
                .expected
                .value
                .as_ref()
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            (
                passed,
                message,
                Some((actual - expected).abs()),
                Some(tolerance()),
            )
        }
        ("array", SpatialObserved::Array1D(actual)) => {
            let expected: Vec<f64> = case
                .expected
                .value
                .as_ref()
                .and_then(|value| serde_json::from_value(value.clone()).ok())
                .unwrap_or_default();
            (
                passed,
                message,
                Some(max_diff_vec(actual, &expected)),
                Some(tolerance()),
            )
        }
        ("matrix", SpatialObserved::Array2D(actual)) => {
            let expected: Vec<Vec<f64>> = case
                .expected
                .value
                .as_ref()
                .and_then(|value| serde_json::from_value(value.clone()).ok())
                .unwrap_or_default();
            (
                passed,
                message,
                Some(max_diff_matrix(actual, &expected)),
                Some(tolerance()),
            )
        }
        (
            "kdtree_query_result",
            SpatialObserved::KdTreeQuery {
                index: actual_index,
                distance: actual_distance,
            },
        ) => {
            let expected_distance = case.expected.distance.unwrap_or(0.0);
            let diff = if *actual_index == case.expected.index.unwrap_or(usize::MAX) {
                (actual_distance - expected_distance).abs()
            } else {
                f64::INFINITY
            };
            (passed, message, Some(diff), Some(tolerance()))
        }
        ("convex_hull", SpatialObserved::ConvexHull { vertices, area }) => {
            let expected_vertices = case.expected.vertices.as_deref().unwrap_or(&[]);
            let expected_area = case.expected.area.unwrap_or(0.0);
            let vertex_diff = if vertices.as_slice() == expected_vertices {
                0.0
            } else {
                f64::INFINITY
            };
            let diff = vertex_diff.max((area - expected_area).abs());
            (passed, message, Some(diff), Some(tolerance()))
        }
        ("procrustes_result", SpatialObserved::Procrustes { disparity }) => {
            let expected_disparity = case.expected.disparity.unwrap_or(0.0);
            (
                passed,
                message,
                Some((disparity - expected_disparity).abs()),
                Some(tolerance()),
            )
        }
        _ => (passed, message, None, None),
    }
}

fn compare_cluster_case_differential(
    case: &ClusterCase,
    observed: &ClusterObserved,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    let (passed, message) = compare_cluster_outcome(case, observed);
    let tolerance = || {
        resolve_cluster_contract_tolerance(
            case.expected.contract_ref.as_deref(),
            case.expected.atol,
            case.expected.rtol,
        )
    };

    match (case.expected.kind.as_str(), observed) {
        ("scalar", ClusterObserved::Scalar(actual)) => {
            let expected = case
                .expected
                .value
                .as_ref()
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            (
                passed,
                message,
                Some((actual - expected).abs()),
                Some(tolerance()),
            )
        }
        ("array", ClusterObserved::Array1D(actual)) => {
            let expected: Vec<f64> = case
                .expected
                .value
                .as_ref()
                .and_then(|value| serde_json::from_value(value.clone()).ok())
                .unwrap_or_default();
            (
                passed,
                message,
                Some(max_diff_vec(actual, &expected)),
                Some(tolerance()),
            )
        }
        ("array" | "matrix", ClusterObserved::Array2D(actual)) => {
            let expected: Vec<Vec<f64>> = case
                .expected
                .value
                .as_ref()
                .and_then(|value| serde_json::from_value(value.clone()).ok())
                .unwrap_or_default();
            (
                passed,
                message,
                Some(max_diff_matrix(actual, &expected)),
                Some(tolerance()),
            )
        }
        ("labels", ClusterObserved::Labels(actual)) => {
            let expected: Vec<usize> = case
                .expected
                .value
                .as_ref()
                .and_then(|value| serde_json::from_value(value.clone()).ok())
                .unwrap_or_default();
            let (expected_cmp, actual_cmp) = if case.expected.deterministic_labels {
                (expected, actual.clone())
            } else {
                (
                    canonicalize_labels_first_occurrence(&expected),
                    canonicalize_labels_first_occurrence(actual),
                )
            };
            let diff = if actual_cmp == expected_cmp {
                0.0
            } else {
                f64::INFINITY
            };
            (passed, message, Some(diff), None)
        }
        ("signed_labels", ClusterObserved::SignedLabels(actual)) => {
            let expected: Vec<i64> = case
                .expected
                .value
                .as_ref()
                .and_then(|value| serde_json::from_value(value.clone()).ok())
                .unwrap_or_default();
            let (expected_cmp, actual_cmp) = if case.expected.deterministic_labels {
                (expected, actual.clone())
            } else {
                (
                    canonicalize_signed_labels_first_occurrence(&expected),
                    canonicalize_signed_labels_first_occurrence(actual),
                )
            };
            let diff = if actual_cmp == expected_cmp {
                0.0
            } else {
                f64::INFINITY
            };
            (passed, message, Some(diff), None)
        }
        ("linkage", ClusterObserved::Linkage(actual)) => {
            let expected: Vec<Vec<f64>> = case
                .expected
                .value
                .as_ref()
                .and_then(|value| serde_json::from_value(value.clone()).ok())
                .unwrap_or_default();
            let actual_matrix: Vec<Vec<f64>> = actual.iter().map(|row| row.to_vec()).collect();
            (
                passed,
                message,
                Some(max_diff_matrix(&actual_matrix, &expected)),
                Some(tolerance()),
            )
        }
        ("vq_result", ClusterObserved::VqResult { codes, dists }) => {
            let expected_codes = case.expected.codes.as_deref().unwrap_or(&[]);
            let expected_dists = case.expected.dists.as_deref().unwrap_or(&[]);
            let code_diff = if codes.as_slice() == expected_codes {
                0.0
            } else {
                f64::INFINITY
            };
            let diff = code_diff.max(max_diff_vec(dists, expected_dists));
            (passed, message, Some(diff), Some(tolerance()))
        }
        ("boolean", ClusterObserved::Boolean(actual)) => {
            let expected = case
                .expected
                .value
                .as_ref()
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            let diff = if *actual == expected {
                0.0
            } else {
                f64::INFINITY
            };
            (passed, message, Some(diff), None)
        }
        _ => (passed, message, None, None),
    }
}

fn compare_casp_case_differential(
    case: &CaspCase,
    observed: &CaspObserved,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    let (passed, message) = compare_casp_outcome(&case.expected, observed);

    match (&case.expected, observed) {
        (CaspExpectedOutcome::PolicyAction { action }, CaspObserved::PolicyAction(actual)) => {
            let diff = if action == actual { 0.0 } else { f64::INFINITY };
            (
                passed,
                message,
                Some(diff),
                Some(ToleranceUsed {
                    atol: 0.0,
                    rtol: 0.0,
                    comparison_mode: "exact".to_owned(),
                }),
            )
        }
        (CaspExpectedOutcome::SolverAction { action }, CaspObserved::SolverAction(actual)) => {
            let diff = if action == actual { 0.0 } else { f64::INFINITY };
            (
                passed,
                message,
                Some(diff),
                Some(ToleranceUsed {
                    atol: 0.0,
                    rtol: 0.0,
                    comparison_mode: "exact".to_owned(),
                }),
            )
        }
        (
            CaspExpectedOutcome::CalibratorFallback { should_fallback },
            CaspObserved::CalibratorFallback(actual),
        ) => {
            let diff = if *should_fallback == *actual {
                0.0
            } else {
                f64::INFINITY
            };
            (
                passed,
                message,
                Some(diff),
                Some(ToleranceUsed {
                    atol: 0.0,
                    rtol: 0.0,
                    comparison_mode: "exact".to_owned(),
                }),
            )
        }
        (CaspExpectedOutcome::Error { error }, CaspObserved::Error(actual)) => {
            let diff = if actual.contains(error) {
                0.0
            } else {
                f64::INFINITY
            };
            (
                passed,
                message,
                Some(diff),
                Some(ToleranceUsed {
                    atol: 0.0,
                    rtol: 0.0,
                    comparison_mode: "substring".to_owned(),
                }),
            )
        }
        _ => (passed, message, None, None),
    }
}

fn compare_fft_case_differential(
    case: &FftCase,
    observed: &FftObserved,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    let (passed, message) = compare_fft_outcome(&case.expected, observed);

    match (&case.expected, observed) {
        (
            FftExpectedOutcome::ComplexVector { values, atol, rtol },
            FftObserved::ComplexVector(actual),
        ) => {
            let diff = actual
                .iter()
                .zip(values.iter())
                .map(|(got, expected)| {
                    (got[0] - expected[0])
                        .abs()
                        .max((got[1] - expected[1]).abs())
                })
                .fold(0.0_f64, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                });
            (
                passed,
                message,
                Some(diff),
                Some(ToleranceUsed {
                    atol: atol.unwrap_or(1.0e-10),
                    rtol: rtol.unwrap_or(1.0e-8),
                    comparison_mode: "allclose".to_owned(),
                }),
            )
        }
        (
            FftExpectedOutcome::RealVector { values, atol, rtol },
            FftObserved::RealVector(actual),
        ) => (
            passed,
            message,
            Some(max_diff_vec(actual, values)),
            Some(ToleranceUsed {
                atol: atol.unwrap_or(1.0e-10),
                rtol: rtol.unwrap_or(1.0e-8),
                comparison_mode: "allclose".to_owned(),
            }),
        ),
        (FftExpectedOutcome::Error { error }, FftObserved::Error(actual)) => {
            let diff = if actual.contains(error) {
                0.0
            } else {
                f64::INFINITY
            };
            (
                passed,
                message,
                Some(diff),
                Some(ToleranceUsed {
                    atol: 0.0,
                    rtol: 0.0,
                    comparison_mode: "substring".to_owned(),
                }),
            )
        }
        _ => (passed, message, None, None),
    }
}

pub fn run_integrate_packet(
    config: &HarnessConfig,
    fixture_name: &str,
) -> Result<PacketReport, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: IntegratePacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path,
            source,
        })?;

    let mut case_results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let observed = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute_integrate_case(case)
        })) {
            Ok(v) => v,
            Err(payload) => {
                case_results.push(CaseResult {
                    case_id: case.case_id().to_owned(),
                    passed: false,
                    message: format!(
                        "PANIC in execute_integrate_case: {}",
                        panic_payload_message(payload)
                    ),
                });
                continue;
            }
        };
        let (passed, message) = compare_integrate_outcome(case, &observed);
        case_results.push(CaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
        });
    }

    Ok(build_packet_report(
        fixture.packet_id,
        fixture.family,
        case_results,
    ))
}

pub fn run_linalg_packet_with_oracle_capture(
    config: &HarnessConfig,
    fixture_name: &str,
    oracle: &PythonOracleConfig,
) -> Result<PacketReport, HarnessError> {
    match capture_linalg_oracle(config, fixture_name, oracle) {
        Ok(output_path) => {
            let capture = load_oracle_capture(&output_path)?;
            let fixture_path = config.fixture_root.join(fixture_name);
            let fixture_raw =
                fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
                    path: fixture_path.clone(),
                    source,
                })?;
            verify_oracle_capture_provenance(&capture, fixture_raw.as_bytes(), fixture_name)?;
            run_linalg_packet_against_oracle_capture(config, fixture_name, &capture)
        }
        Err(err) => {
            if oracle.required {
                return Err(err);
            }

            let report = run_linalg_packet(config, fixture_name)?;
            let failure_path = config
                .artifact_dir_for(&report.packet_id)
                .join("oracle_capture.error.txt");
            fs::create_dir_all(config.artifact_dir_for(&report.packet_id)).map_err(|source| {
                HarnessError::ArtifactIo {
                    path: config.artifact_dir_for(&report.packet_id),
                    source,
                }
            })?;
            fs::write(&failure_path, format!("{err}")).map_err(|source| {
                HarnessError::ArtifactIo {
                    path: failure_path,
                    source,
                }
            })?;

            Ok(report)
        }
    }
}

pub fn capture_linalg_oracle(
    config: &HarnessConfig,
    fixture_name: &str,
    oracle: &PythonOracleConfig,
) -> Result<PathBuf, HarnessError> {
    let fixture_path = config.fixture_root.join(fixture_name);
    let raw = fs::read_to_string(&fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.clone(),
        source,
    })?;
    let fixture: LinalgPacketFixture =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.clone(),
            source,
        })?;

    let output_dir = config.artifact_dir_for(&fixture.packet_id);
    fs::create_dir_all(&output_dir).map_err(|source| HarnessError::ArtifactIo {
        path: output_dir.clone(),
        source,
    })?;
    let output_path = output_dir.join("oracle_capture.json");

    if !oracle.script_path.exists() {
        return Err(HarnessError::PythonScriptMissing {
            path: oracle.script_path.clone(),
        });
    }

    let python_bin = oracle.python_bin.display().to_string();
    let output = Command::new(&oracle.python_bin)
        .arg(&oracle.script_path)
        .arg(as_os_str("--fixture"))
        .arg(&fixture_path)
        .arg(as_os_str("--output"))
        .arg(&output_path)
        .arg(as_os_str("--oracle-root"))
        .arg(&config.oracle_root)
        .output()
        .map_err(|source| HarnessError::PythonLaunch {
            python_bin: python_bin.clone(),
            source,
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        return Err(classify_python_stderr(python_bin, stderr));
    }

    let mut parsed = load_oracle_capture(&output_path)?;
    attach_oracle_capture_provenance(&mut parsed, raw.as_bytes())?;
    let normalized =
        serde_json::to_vec_pretty(&parsed).map_err(|e| HarnessError::RaptorQ(e.to_string()))?;
    fs::write(&output_path, normalized).map_err(|source| HarnessError::ArtifactIo {
        path: output_path.clone(),
        source,
    })?;

    Ok(output_path)
}

pub fn load_oracle_capture(path: &Path) -> Result<OracleCapture, HarnessError> {
    let raw = fs::read_to_string(path).map_err(|source| HarnessError::FixtureIo {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&raw).map_err(|source| HarnessError::OracleParse {
        path: path.to_path_buf(),
        source,
    })
}

/// Verify that a loaded oracle capture's provenance hashes still match the
/// current fixture bytes. If the fixture has been edited since the oracle
/// was generated (new cases, changed args, reworded typos), the stored
/// `capture.provenance.fixture_input_blake3` will no longer equal
/// blake3(fixture_input) and we emit an OracleCaptureMismatch rather than
/// silently comparing Rust outputs against stale oracle values.
///
/// No-op when the capture has no recorded provenance (legacy captures
/// generated before attach_oracle_capture_provenance existed). Once all
/// captures are regenerated this relaxation can be tightened to require
/// provenance. Tracked via frankenscipy-cpn9.
fn verify_oracle_capture_provenance(
    capture: &OracleCapture,
    fixture_input: &[u8],
    fixture_label: &str,
) -> Result<(), HarnessError> {
    let Some(provenance) = capture.provenance.as_ref() else {
        return Ok(());
    };
    let current_fixture_hash = hash(fixture_input).to_hex().to_string();
    if current_fixture_hash != provenance.fixture_input_blake3 {
        return Err(HarnessError::OracleCaptureMismatch {
            detail: format!(
                "stale oracle capture for {fixture_label}: fixture input blake3 differs \
                 (capture recorded {} but fixture now hashes to {}). \
                 Re-run the oracle capture script to regenerate.",
                provenance.fixture_input_blake3, current_fixture_hash
            ),
        });
    }
    Ok(())
}

fn attach_oracle_capture_provenance(
    capture: &mut OracleCapture,
    fixture_input: &[u8],
) -> Result<(), HarnessError> {
    let oracle_output = serde_json::to_vec(&capture.case_outputs)
        .map_err(|e| HarnessError::RaptorQ(e.to_string()))?;
    let fixture_input_blake3 = hash(fixture_input).to_hex().to_string();
    let oracle_output_blake3 = hash(&oracle_output).to_hex().to_string();
    let mut capture_hasher = blake3::Hasher::new();
    if let Some(runtime) = &capture.runtime {
        capture_hasher.update(runtime.python_version.as_bytes());
        capture_hasher.update(runtime.numpy_version.as_bytes());
        capture_hasher.update(runtime.scipy_version.as_bytes());
    }
    capture_hasher.update(fixture_input);
    capture_hasher.update(&oracle_output);
    capture.provenance = Some(OracleCaptureProvenance {
        fixture_input_blake3,
        oracle_output_blake3,
        capture_blake3: capture_hasher.finalize().to_hex().to_string(),
    });
    Ok(())
}

fn oracle_result_field<T: DeserializeOwned>(
    result: &serde_json::Value,
    field: &str,
    case_id: &str,
) -> Result<T, String> {
    let value = result
        .get(field)
        .cloned()
        .ok_or_else(|| format!("oracle result for {case_id} missing field `{field}`"))?;
    serde_json::from_value(value)
        .map_err(|e| format!("oracle result for {case_id} has invalid `{field}`: {e}"))
}

fn linalg_expected_tolerance(expected: &LinalgExpectedOutcome) -> Option<(f64, f64)> {
    match expected {
        LinalgExpectedOutcome::Vector { atol, rtol, .. }
        | LinalgExpectedOutcome::Matrix { atol, rtol, .. }
        | LinalgExpectedOutcome::Scalar { atol, rtol, .. }
        | LinalgExpectedOutcome::Lstsq { atol, rtol, .. }
        | LinalgExpectedOutcome::Pinv { atol, rtol, .. } => Some((*atol, *rtol)),
        LinalgExpectedOutcome::Error { .. } => None,
    }
}

fn linalg_expected_warning(expected: &LinalgExpectedOutcome) -> Option<bool> {
    match expected {
        LinalgExpectedOutcome::Vector {
            expect_warning_ill_conditioned,
            ..
        } => *expect_warning_ill_conditioned,
        _ => None,
    }
}

fn oracle_case_to_expected(
    case: &LinalgCase,
    oracle_case: &OracleCaseOutput,
) -> Result<LinalgExpectedOutcome, String> {
    let (atol, rtol) = linalg_expected_tolerance(case.expected()).ok_or_else(|| {
        format!(
            "case {} cannot use oracle numeric comparison for error-only expectation",
            case.case_id()
        )
    })?;

    match oracle_case.result_kind.as_str() {
        "vector" => Ok(LinalgExpectedOutcome::Vector {
            values: oracle_result_field(&oracle_case.result, "values", case.case_id())?,
            atol,
            rtol,
            expect_warning_ill_conditioned: linalg_expected_warning(case.expected()),
        }),
        "matrix" => Ok(LinalgExpectedOutcome::Matrix {
            values: oracle_result_field(&oracle_case.result, "values", case.case_id())?,
            atol,
            rtol,
        }),
        "scalar" => Ok(LinalgExpectedOutcome::Scalar {
            value: oracle_result_field(&oracle_case.result, "value", case.case_id())?,
            atol,
            rtol,
        }),
        "lstsq" => Ok(LinalgExpectedOutcome::Lstsq {
            x: oracle_result_field(&oracle_case.result, "x", case.case_id())?,
            residuals: oracle_result_field(&oracle_case.result, "residuals", case.case_id())?,
            rank: oracle_result_field(&oracle_case.result, "rank", case.case_id())?,
            singular_values: oracle_result_field(
                &oracle_case.result,
                "singular_values",
                case.case_id(),
            )?,
            atol,
            rtol,
        }),
        "pinv" => Ok(LinalgExpectedOutcome::Pinv {
            values: oracle_result_field(&oracle_case.result, "values", case.case_id())?,
            rank: oracle_result_field(&oracle_case.result, "rank", case.case_id())?,
            atol,
            rtol,
        }),
        other => Err(format!(
            "oracle result for {} has unsupported result_kind `{other}`",
            case.case_id()
        )),
    }
}

fn compare_linalg_case_against_oracle(
    case: &LinalgCase,
    oracle_case: &OracleCaseOutput,
    observed: &Result<LinalgObservedOutcome, LinalgError>,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    if matches!(case.expected(), LinalgExpectedOutcome::Error { .. }) {
        return compare_linalg_case_differential(case.expected(), observed);
    }

    if oracle_case.status != "ok" {
        return match observed {
            Err(actual) => (
                false,
                format!(
                    "oracle errored (`{}`) and rust errored (`{actual}`) too; case is unjudgeable",
                    oracle_case
                        .error
                        .as_deref()
                        .unwrap_or("unknown oracle error"),
                ),
                None,
                None,
            ),
            Ok(_) => (
                false,
                format!(
                    "oracle errored (`{}`) but rust succeeded",
                    oracle_case
                        .error
                        .as_deref()
                        .unwrap_or("unknown oracle error"),
                ),
                None,
                None,
            ),
        };
    }

    match oracle_case_to_expected(case, oracle_case) {
        Ok(expected) => compare_linalg_case_differential(&expected, observed),
        Err(message) => (false, message, None, None),
    }
}

fn stats_expected_tolerance(expected: &StatsExpected) -> (f64, f64) {
    (
        expected.atol.unwrap_or(1e-10),
        expected.rtol.unwrap_or(1e-10),
    )
}

fn stats_oracle_case_to_expected(
    case: &StatsCase,
    oracle_case: &OracleCaseOutput,
) -> Result<StatsExpected, String> {
    let (atol, rtol) = stats_expected_tolerance(&case.expected);
    let contract_ref = case.expected.contract_ref.clone();

    let mut expected = StatsExpected {
        kind: oracle_case.result_kind.clone(),
        value: None,
        nobs: None,
        minmax: None,
        mean: None,
        variance: None,
        skewness: None,
        kurtosis: None,
        statistic: None,
        pvalue: None,
        slope: None,
        intercept: None,
        rvalue: None,
        stderr: None,
        array_value: None,
        atol: Some(atol),
        rtol: Some(rtol),
        contract_ref,
    };

    match oracle_case.result_kind.as_str() {
        "scalar" => {
            expected.value = Some(oracle_result_field(
                &oracle_case.result,
                "value",
                case.case_id(),
            )?);
        }
        "array" => {
            expected.array_value = Some(oracle_result_field(
                &oracle_case.result,
                "values",
                case.case_id(),
            )?);
        }
        "describe_result" => {
            expected.nobs = Some(oracle_result_field(
                &oracle_case.result,
                "nobs",
                case.case_id(),
            )?);
            expected.minmax = Some(oracle_result_field(
                &oracle_case.result,
                "minmax",
                case.case_id(),
            )?);
            expected.mean = Some(oracle_result_field(
                &oracle_case.result,
                "mean",
                case.case_id(),
            )?);
            expected.variance = Some(oracle_result_field(
                &oracle_case.result,
                "variance",
                case.case_id(),
            )?);
            expected.skewness = Some(oracle_result_field(
                &oracle_case.result,
                "skewness",
                case.case_id(),
            )?);
            expected.kurtosis = Some(oracle_result_field(
                &oracle_case.result,
                "kurtosis",
                case.case_id(),
            )?);
        }
        "correlation_result" | "ttest_result" | "goodness_result" => {
            expected.statistic = Some(oracle_result_field(
                &oracle_case.result,
                "statistic",
                case.case_id(),
            )?);
            expected.pvalue = Some(oracle_result_field(
                &oracle_case.result,
                "pvalue",
                case.case_id(),
            )?);
        }
        "linregress_result" => {
            expected.slope = Some(oracle_result_field(
                &oracle_case.result,
                "slope",
                case.case_id(),
            )?);
            expected.intercept = Some(oracle_result_field(
                &oracle_case.result,
                "intercept",
                case.case_id(),
            )?);
            expected.rvalue = Some(oracle_result_field(
                &oracle_case.result,
                "rvalue",
                case.case_id(),
            )?);
            expected.pvalue = Some(oracle_result_field(
                &oracle_case.result,
                "pvalue",
                case.case_id(),
            )?);
            expected.stderr = Some(oracle_result_field(
                &oracle_case.result,
                "stderr",
                case.case_id(),
            )?);
        }
        other => {
            return Err(format!(
                "oracle result for {} has unsupported result_kind `{other}`",
                case.case_id()
            ));
        }
    }

    Ok(expected)
}

fn compare_stats_case_against_oracle(
    case: &StatsCase,
    oracle_case: &OracleCaseOutput,
    observed: &StatsObserved,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    if oracle_case.status != "ok" {
        // br-7vhu: result_kind "skipped" means the oracle deliberately
        // declined this case (e.g. bootstrap_mean PRNG mismatch). Fall
        // through to the fixture-expected comparison instead of failing.
        if oracle_case.result_kind == "skipped" {
            return compare_stats_case_differential(case, observed);
        }
        return match observed {
            StatsObserved::Error(actual) => (
                false,
                format!(
                    "oracle errored (`{}`) and rust errored (`{actual}`) too; case is unjudgeable",
                    oracle_case
                        .error
                        .as_deref()
                        .unwrap_or("unknown oracle error"),
                ),
                None,
                None,
            ),
            _ => (
                false,
                format!(
                    "oracle errored (`{}`) but rust succeeded",
                    oracle_case
                        .error
                        .as_deref()
                        .unwrap_or("unknown oracle error"),
                ),
                None,
                None,
            ),
        };
    }

    match stats_oracle_case_to_expected(case, oracle_case) {
        Ok(expected) => {
            let mut oracle_case_fixture = case.clone();
            oracle_case_fixture.expected = expected;
            compare_stats_case_differential(&oracle_case_fixture, observed)
        }
        Err(message) => (false, message, None, None),
    }
}

fn oracle_status_from_capture_error(error: &HarnessError) -> OracleStatus {
    match error {
        HarnessError::PythonScriptMissing { path } => OracleStatus::Missing {
            reason: format!("script not found: {}", path.display()),
        },
        HarnessError::PythonSciPyMissing { stderr } => OracleStatus::Missing {
            reason: format!("scipy not available: {stderr}"),
        },
        HarnessError::PythonLaunch { python_bin, source } => OracleStatus::Missing {
            reason: format!("failed to launch python oracle `{python_bin}`: {source}"),
        },
        HarnessError::PythonFailed { stderr, .. } => OracleStatus::Failed {
            reason: stderr.clone(),
        },
        HarnessError::OracleParse { path, source } => OracleStatus::Failed {
            reason: format!(
                "oracle capture parse failed for {}: {source}",
                path.display()
            ),
        },
        other => OracleStatus::Failed {
            reason: other.to_string(),
        },
    }
}

fn default_differential_oracle_script_path(family: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // Route by family substring to the corresponding oracle script.
    // Ordered longest/most-specific first so e.g. "array_api" never matches
    // a hypothetical shorter token. Families without a dedicated script
    // retain the sentinel default so probes fail closed instead of silently
    // running the wrong oracle.
    let script_name = if family.contains("array_api") || family.contains("arrayapi") {
        "scipy_arrayapi_oracle.py"
    } else if family.contains("constants") {
        "scipy_constants_oracle.py"
    } else if family.contains("io") {
        "scipy_io_oracle.py"
    } else if family.contains("stats") {
        "scipy_stats_oracle.py"
    } else if family.contains("interpolate") {
        "scipy_interpolate_oracle.py"
    } else if family.contains("ndimage") {
        "scipy_ndimage_oracle.py"
    } else if family.contains("cluster") {
        "scipy_cluster_oracle.py"
    } else if family.contains("spatial") {
        "scipy_spatial_oracle.py"
    } else if family.contains("signal") {
        "scipy_signal_oracle.py"
    } else if family.contains("integrate") {
        "scipy_integrate_oracle.py"
    } else if family.contains("optimize") {
        "scipy_optimize_oracle.py"
    } else if family.contains("sparse") {
        "scipy_sparse_oracle.py"
    } else if family.contains("special") {
        "scipy_special_oracle.py"
    } else if family.contains("fft") {
        "scipy_fft_oracle.py"
    } else if family.contains("linalg") {
        "scipy_linalg_oracle.py"
    } else {
        return DifferentialOracleConfig::default().script_path;
    };
    manifest.join("python_oracle").join(script_name)
}

fn resolve_differential_oracle_config(
    config: &DifferentialOracleConfig,
    family: &str,
) -> DifferentialOracleConfig {
    let default_script_path = DifferentialOracleConfig::default().script_path;
    let script_path = if config.script_path == default_script_path {
        default_differential_oracle_script_path(family)
    } else {
        config.script_path.clone()
    };

    DifferentialOracleConfig {
        python_path: config.python_path.clone(),
        script_path,
        timeout_secs: config.timeout_secs,
        required: config.required,
    }
}

fn capture_python_oracle_inner(
    fixture_path: &Path,
    fixture_raw: &str,
    packet_id: &str,
    oracle_root: &Path,
    oracle: &DifferentialOracleConfig,
) -> Result<OracleCapture, HarnessError> {
    let output_dir = differential_artifact_dir_for_fixture(fixture_path, packet_id);
    fs::create_dir_all(&output_dir).map_err(|source| HarnessError::ArtifactIo {
        path: output_dir.clone(),
        source,
    })?;
    let output_path = output_dir.join("oracle_capture.json");

    if !oracle.script_path.exists() {
        return Err(HarnessError::PythonScriptMissing {
            path: oracle.script_path.clone(),
        });
    }

    let python_bin = oracle.python_path.display().to_string();
    let output = Command::new(&oracle.python_path)
        .arg(&oracle.script_path)
        .arg(as_os_str("--fixture"))
        .arg(fixture_path)
        .arg(as_os_str("--output"))
        .arg(&output_path)
        .arg(as_os_str("--oracle-root"))
        .arg(oracle_root)
        .output()
        .map_err(|source| HarnessError::PythonLaunch {
            python_bin: python_bin.clone(),
            source,
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        return Err(classify_python_stderr(python_bin, stderr));
    }

    let mut parsed = load_oracle_capture(&output_path)?;
    attach_oracle_capture_provenance(&mut parsed, fixture_raw.as_bytes())?;
    let normalized =
        serde_json::to_vec_pretty(&parsed).map_err(|e| HarnessError::RaptorQ(e.to_string()))?;
    fs::write(&output_path, normalized).map_err(|source| HarnessError::ArtifactIo {
        path: output_path,
        source,
    })?;

    Ok(parsed)
}

pub fn load_packet_reports(config: &HarnessConfig) -> Result<Vec<PacketReport>, HarnessError> {
    let artifact_root = config.fixture_root.join("artifacts");
    if !artifact_root.exists() {
        return Ok(Vec::new());
    }

    let mut reports = Vec::new();
    for packet_dir in fs::read_dir(&artifact_root).map_err(|source| HarnessError::ArtifactIo {
        path: artifact_root.clone(),
        source,
    })? {
        let packet_dir = packet_dir
            .map_err(|source| HarnessError::ArtifactIo {
                path: artifact_root.clone(),
                source,
            })?
            .path();
        if !packet_dir.is_dir() {
            continue;
        }
        let report_path = packet_dir.join("parity_report.json");
        if !report_path.exists() {
            continue;
        }
        let raw = fs::read_to_string(&report_path).map_err(|source| HarnessError::ArtifactIo {
            path: report_path.clone(),
            source,
        })?;
        let report: PacketReport =
            serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
                path: report_path,
                source,
            })?;
        reports.push(report);
    }

    reports.sort_by(|a, b| a.packet_id.cmp(&b.packet_id));
    Ok(reports)
}

/// Serializes concurrent writers of the parity-report triple
/// (parity_report.json + .raptorq.json + .decode_proof.json) so that
/// multi-test runs targeting the same packet output dir cannot
/// interleave their writes and leave the sidecar pointing at a hash
/// that no longer matches the on-disk report (frankenscipy-prngc).
fn packet_report_write_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn write_packet_report_artifacts(
    output_dir: &Path,
    report: &PacketReport,
) -> Result<ParityArtifactBundle, HarnessError> {
    fs::create_dir_all(output_dir).map_err(|source| HarnessError::ArtifactIo {
        path: output_dir.to_path_buf(),
        source,
    })?;

    // Hold the global writer mutex for the duration of the report +
    // sidecar + decode-proof writes so the trio is always coherent
    // even when cargo test schedules many P2C-* differential tests in
    // parallel against the same output dir.
    let _guard = packet_report_write_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    let report_path = output_dir.join("parity_report.json");
    // /testing-golden-artifacts for [frankenscipy-okuhi]: stability —
    // if a parity_report.json already exists and its content matches
    // the new report modulo `generated_unix_ms`, reuse the on-disk
    // bytes verbatim. The .raptorq.json and .decode_proof.json
    // sidecars are hash-derived from these bytes, so preserving them
    // unchanged transitively stabilises the whole bundle. Without
    // this check, every test run would rewrite three committed files
    // with no semantic difference, forcing periodic "refresh parity
    // reports" churn commits.
    let new_bytes =
        serde_json::to_vec_pretty(report).map_err(|e| HarnessError::RaptorQ(e.to_string()))?;
    let report_bytes = match fs::read(&report_path) {
        Ok(existing) => match serde_json::from_slice::<PacketReport>(&existing) {
            Ok(existing_report) => {
                let mut probe = report.clone();
                probe.generated_unix_ms = existing_report.generated_unix_ms;
                if probe == existing_report {
                    existing
                } else {
                    new_bytes
                }
            }
            Err(_) => new_bytes,
        },
        Err(_) => new_bytes,
    };
    fs::write(&report_path, &report_bytes).map_err(|source| HarnessError::ArtifactIo {
        path: report_path.clone(),
        source,
    })?;

    let sidecar = generate_raptorq_sidecar(&report_bytes)?;
    let sidecar_path = output_dir.join("parity_report.raptorq.json");
    let sidecar_bytes =
        serde_json::to_vec_pretty(&sidecar).map_err(|e| HarnessError::RaptorQ(e.to_string()))?;
    fs::write(&sidecar_path, sidecar_bytes).map_err(|source| HarnessError::ArtifactIo {
        path: sidecar_path.clone(),
        source,
    })?;

    let decode_proof = generate_decode_proof_artifact(&report_bytes, &sidecar)?;
    let decode_proof_path = output_dir.join("parity_report.decode_proof.json");
    // /testing-golden-artifacts for [frankenscipy-okuhi]: same
    // content-stability rule as parity_report.json — if the on-disk
    // decode proof matches the freshly-generated one modulo
    // `ts_unix_ms`, keep the on-disk bytes verbatim.
    let new_proof_bytes = serde_json::to_vec_pretty(&decode_proof)
        .map_err(|e| HarnessError::RaptorQ(e.to_string()))?;
    let decode_proof_bytes = match fs::read(&decode_proof_path) {
        Ok(existing) => match serde_json::from_slice::<DecodeProofArtifact>(&existing) {
            Ok(existing_proof) => {
                let mut probe = decode_proof.clone();
                probe.ts_unix_ms = existing_proof.ts_unix_ms;
                if probe == existing_proof {
                    existing
                } else {
                    new_proof_bytes
                }
            }
            Err(_) => new_proof_bytes,
        },
        Err(_) => new_proof_bytes,
    };
    fs::write(&decode_proof_path, decode_proof_bytes).map_err(|source| {
        HarnessError::ArtifactIo {
            path: decode_proof_path.clone(),
            source,
        }
    })?;

    Ok(ParityArtifactBundle {
        report_path,
        sidecar_path,
        decode_proof_path,
    })
}

fn hex_decode(input: &str) -> Result<Vec<u8>, HarnessError> {
    if !input.len().is_multiple_of(2) {
        return Err(HarnessError::RaptorQ(
            "repair symbol payload hex must have even length".to_owned(),
        ));
    }

    let mut out = Vec::with_capacity(input.len() / 2);
    let bytes = input.as_bytes();
    let hex_val = |byte: u8| -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(10 + (byte - b'a')),
            b'A'..=b'F' => Some(10 + (byte - b'A')),
            _ => None,
        }
    };

    let mut idx = 0;
    while idx < bytes.len() {
        let hi = hex_val(bytes[idx]).ok_or_else(|| {
            HarnessError::RaptorQ(format!(
                "repair symbol payload contains invalid hex nibble at byte {idx}"
            ))
        })?;
        let lo = hex_val(bytes[idx + 1]).ok_or_else(|| {
            HarnessError::RaptorQ(format!(
                "repair symbol payload contains invalid hex nibble at byte {}",
                idx + 1
            ))
        })?;
        out.push((hi << 4) | lo);
        idx += 2;
    }

    Ok(out)
}

fn recover_payload_with_sidecar(
    payload: &[u8],
    sidecar: &RaptorQSidecar,
    drop_indices: &[usize],
) -> Result<Vec<u8>, HarnessError> {
    // Reject zero symbol_size up front. RaptorQSidecar is serde-derived
    // and may be loaded from an external JSON (raptorq_sidecar --verify),
    // so a malformed/tampered file with symbol_size=0 would otherwise
    // panic in chunk_payload (division by zero in div_ceil and chunks).
    // Review-mode finding: typed error is the right surface here.
    if sidecar.symbol_size == 0 {
        return Err(HarnessError::RaptorQ(
            "sidecar symbol_size must be non-zero".to_owned(),
        ));
    }
    let source_symbols = chunk_payload(payload, sidecar.symbol_size);
    let k = source_symbols.len();
    if k != sidecar.source_symbols {
        return Err(HarnessError::RaptorQ(format!(
            "sidecar source-symbol mismatch: expected {}, got {}",
            sidecar.source_symbols, k
        )));
    }
    if sidecar.repair_symbol_payloads_hex.len() != sidecar.repair_symbols {
        return Err(HarnessError::RaptorQ(format!(
            "repair payload count mismatch: expected {}, got {}",
            sidecar.repair_symbols,
            sidecar.repair_symbol_payloads_hex.len()
        )));
    }

    // Fail loud on out-of-range drop_indices instead of silently
    // skipping. Previously `if idx < k { dropped[idx] = true }`
    // swallowed stale indices, so a caller passing a wrong/stale
    // index would get a "decode proof" that didn't actually exercise
    // recovery — review-mode finding.
    let mut dropped = vec![false; k];
    for &idx in drop_indices {
        if idx >= k {
            return Err(HarnessError::RaptorQ(format!(
                "drop_indices contains out-of-range symbol index {idx} (k = {k})"
            )));
        }
        dropped[idx] = true;
    }

    let mut received = Vec::new();
    for (idx, symbol) in source_symbols.iter().enumerate() {
        if !dropped[idx] {
            received.push(ReceivedSymbol::source(idx as u32, symbol.clone()));
        }
    }

    let decoder = InactivationDecoder::new(k, sidecar.symbol_size, sidecar.seed);
    for (offset, payload_hex) in sidecar.repair_symbol_payloads_hex.iter().enumerate() {
        let esi = k as u32 + offset as u32;
        let payload = hex_decode(payload_hex)?;
        let (cols, coefs) = decoder.repair_equation(esi);
        received.push(ReceivedSymbol::repair(esi, cols, coefs, payload));
    }

    received.extend(decoder.constraint_symbols());

    let result = decoder
        .decode(&received)
        .map_err(|err| HarnessError::RaptorQ(format!("decode proof recovery failed: {err:?}")))?;

    let mut recovered = Vec::with_capacity(payload.len());
    for symbol in result.source.iter().take(k) {
        recovered.extend_from_slice(symbol);
    }
    recovered.truncate(payload.len());
    Ok(recovered)
}

pub fn generate_decode_proof_artifact(
    payload: &[u8],
    sidecar: &RaptorQSidecar,
) -> Result<DecodeProofArtifact, HarnessError> {
    // Single source of truth for which symbols we drop and how many
    // we recover. Previously the count was a magic constant (= 1)
    // separately defined from the drop_indices slice; a future refactor
    // bumping the drops without the count would silently produce a
    // misleading DecodeProofArtifact. Review-mode finding [byfq8].
    let drop_indices: &[usize] = &[0];
    let recovered_blocks = drop_indices.len();
    let recovered = recover_payload_with_sidecar(payload, sidecar, drop_indices)?;
    let proof_hash = hash(&recovered).to_hex().to_string();
    let expected_hash = hash(payload).to_hex().to_string();
    if proof_hash != expected_hash {
        return Err(HarnessError::RaptorQ(
            "decode proof hash mismatch after simulated recovery".to_owned(),
        ));
    }

    Ok(DecodeProofArtifact {
        ts_unix_ms: now_unix_ms(),
        reason: format!(
            "recovered from simulated corruption of {recovered_blocks} block{}",
            if recovered_blocks == 1 { "" } else { "s" },
        ),
        recovered_blocks,
        proof_hash,
    })
}

#[must_use]
pub fn packet_summary(report: &PacketReport) -> PacketSummary {
    PacketSummary {
        packet_id: report.packet_id.clone(),
        family: report.family.clone(),
        passed_cases: report.passed_cases,
        failed_cases: report.failed_cases,
        total_cases: report.case_results.len(),
    }
}

pub fn write_parity_artifacts(
    config: &HarnessConfig,
    report: &PacketReport,
) -> Result<ParityArtifactBundle, HarnessError> {
    let output_dir = config.artifact_dir_for(&report.packet_id);
    write_packet_report_artifacts(&output_dir, report)
}

pub fn write_differential_parity_artifacts(
    fixture_path: &Path,
    report: &ConformanceReport,
) -> Result<ParityArtifactBundle, HarnessError> {
    let output_dir = differential_artifact_dir_for_fixture(fixture_path, &report.packet_id);
    let packet_report = PacketReport::from(report);
    write_packet_report_artifacts(&output_dir, &packet_report)
}

#[must_use]
pub fn build_drift_diff_report(report: &ConformanceReport) -> DriftDiffReport {
    let cases = report
        .per_case_results
        .iter()
        .map(|case| DriftCaseDiff {
            case_id: case.case_id.clone(),
            passed: case.passed,
            message: case.message.clone(),
            max_diff: case.max_diff,
            tolerance_used: case.tolerance_used.clone(),
            oracle_status: case.oracle_status.clone(),
        })
        .collect::<Vec<_>>();
    let drifted_cases = cases.iter().filter(|case| !case.passed).count();

    DriftDiffReport {
        schema_version: 1,
        packet_id: report.packet_id.clone(),
        family: report.family.clone(),
        fixture_path: report.fixture_path.clone(),
        generated_unix_ms: report.generated_unix_ms,
        oracle_status: report.oracle_status.clone(),
        total_cases: report.per_case_results.len(),
        drifted_cases,
        cases,
    }
}

pub fn write_drift_diff_artifact(
    fixture_path: &Path,
    drift: &DriftDiffReport,
) -> Result<PathBuf, HarnessError> {
    let output_dir = differential_artifact_dir_for_fixture(fixture_path, &drift.packet_id);
    fs::create_dir_all(&output_dir).map_err(|source| HarnessError::ArtifactIo {
        path: output_dir.clone(),
        source,
    })?;

    let path = output_dir.join("drift_diff.json");
    let bytes =
        serde_json::to_vec_pretty(drift).map_err(|e| HarnessError::Serialization(e.to_string()))?;
    fs::write(&path, bytes).map_err(|source| HarnessError::ArtifactIo {
        path: path.clone(),
        source,
    })?;
    Ok(path)
}

pub fn discover_differential_fixture_paths(
    config: &HarnessConfig,
) -> Result<Vec<PathBuf>, HarnessError> {
    let mut fixtures = Vec::new();
    let entries = fs::read_dir(&config.fixture_root).map_err(|source| HarnessError::FixtureIo {
        path: config.fixture_root.clone(),
        source,
    })?;

    for entry in entries {
        let entry = entry.map_err(|source| HarnessError::FixtureIo {
            path: config.fixture_root.clone(),
            source,
        })?;
        let path = entry.path();
        if !is_p2c_fixture_path(&path) {
            continue;
        }
        let raw = fs::read_to_string(&path).map_err(|source| HarnessError::FixtureIo {
            path: path.clone(),
            source,
        })?;
        let _: FixtureEnvelope =
            serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
                path: path.clone(),
                source,
            })?;
        fixtures.push(path);
    }

    fixtures.sort();
    Ok(fixtures)
}

pub fn run_live_oracle_capture_lane(
    config: &HarnessConfig,
    oracle_config: &DifferentialOracleConfig,
    packet_filter: Option<&str>,
) -> Result<LiveOracleCaptureReport, HarnessError> {
    let packet_filter = packet_filter.map(str::to_owned);
    let fixtures = discover_differential_fixture_paths(config)?;
    let mut packets = Vec::new();

    for fixture_path in fixtures {
        if !matches_packet_filter(&fixture_path, packet_filter.as_deref()) {
            continue;
        }
        let report = run_differential_test(&fixture_path, oracle_config)?;
        let drift = build_drift_diff_report(&report);
        let _ = write_drift_diff_artifact(&fixture_path, &drift)?;
        packets.push(drift);
    }

    let total_cases = packets.iter().map(|packet| packet.total_cases).sum();
    let drifted_cases = packets.iter().map(|packet| packet.drifted_cases).sum();
    let oracle_available_packets = packets
        .iter()
        .filter(|packet| matches!(packet.oracle_status, OracleStatus::Available))
        .count();
    let oracle_unavailable_packets = packets.len().saturating_sub(oracle_available_packets);

    Ok(LiveOracleCaptureReport {
        schema_version: 1,
        generated_unix_ms: now_unix_ms(),
        fixture_root: config.fixture_root.display().to_string(),
        total_packets: packets.len(),
        total_cases,
        drifted_cases,
        oracle_available_packets,
        oracle_unavailable_packets,
        packets,
    })
}

fn is_p2c_fixture_path(path: &Path) -> bool {
    path.extension().and_then(OsStr::to_str) == Some("json")
        && path
            .file_name()
            .and_then(OsStr::to_str)
            .is_some_and(|name| name.starts_with("FSCI-P2C-"))
}

fn matches_packet_filter(path: &Path, packet_filter: Option<&str>) -> bool {
    let Some(filter) = packet_filter else {
        return true;
    };
    path.file_stem()
        .and_then(OsStr::to_str)
        .is_some_and(|stem| stem.contains(filter))
}

#[cfg(feature = "dashboard")]
#[must_use]
pub fn style_for_case_result(case: &CaseResult) -> Style {
    if case.passed {
        Style::new().fg(PackedRgba::rgb(48, 186, 95)).bold()
    } else {
        Style::new().fg(PackedRgba::rgb(216, 63, 63)).bold()
    }
}

#[cfg(feature = "dashboard")]
#[must_use]
pub fn style_for_packet_summary(summary: &PacketSummary) -> Style {
    if summary.failed_cases == 0 {
        Style::new().fg(PackedRgba::rgb(70, 200, 120)).bold()
    } else {
        Style::new().fg(PackedRgba::rgb(220, 90, 70)).bold()
    }
}

fn execute_linalg_case(case: &LinalgCase) -> Result<LinalgObservedOutcome, LinalgError> {
    match case {
        LinalgCase::Solve {
            mode,
            a,
            b,
            assume_a,
            lower,
            transposed,
            check_finite,
            ..
        } => {
            let result = solve(
                a,
                b,
                SolveOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                    assume_a: assume_a.clone().map(Into::into),
                    lower: lower.unwrap_or(false),
                    transposed: transposed.unwrap_or(false),
                },
            )?;
            Ok(LinalgObservedOutcome::Vector {
                values: result.x,
                warning_ill_conditioned: result.warning.is_some(),
            })
        }
        LinalgCase::SolveTriangular {
            mode,
            a,
            b,
            trans,
            lower,
            unit_diagonal,
            check_finite,
            ..
        } => {
            let result = solve_triangular(
                a,
                b,
                TriangularSolveOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                    trans: trans
                        .clone()
                        .unwrap_or(FixtureTriangularTranspose::NoTranspose)
                        .into(),
                    lower: lower.unwrap_or(false),
                    unit_diagonal: unit_diagonal.unwrap_or(false),
                },
            )?;
            Ok(LinalgObservedOutcome::Vector {
                values: result.x,
                warning_ill_conditioned: false,
            })
        }
        LinalgCase::SolveBanded {
            mode,
            l_and_u,
            ab,
            b,
            check_finite,
            ..
        } => {
            let result = solve_banded(
                (l_and_u[0], l_and_u[1]),
                ab,
                b,
                SolveOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                    assume_a: Some(MatrixAssumption::General),
                    lower: false,
                    transposed: false,
                },
            )?;
            Ok(LinalgObservedOutcome::Vector {
                values: result.x,
                warning_ill_conditioned: result.warning.is_some(),
            })
        }
        LinalgCase::Inv {
            mode,
            a,
            assume_a,
            lower,
            check_finite,
            ..
        } => {
            let result = inv(
                a,
                InvOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                    assume_a: assume_a.clone().map(Into::into),
                    lower: lower.unwrap_or(false),
                },
            )?;
            Ok(LinalgObservedOutcome::Matrix {
                values: result.inverse,
            })
        }
        LinalgCase::Det {
            mode,
            a,
            check_finite,
            ..
        } => {
            let value = det(a, *mode, check_finite.unwrap_or(true))?;
            Ok(LinalgObservedOutcome::Scalar { value })
        }
        LinalgCase::Lstsq {
            mode,
            a,
            b,
            cond,
            check_finite,
            ..
        } => {
            let result = lstsq(
                a,
                b,
                LstsqOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                    cond: *cond,
                    driver: LstsqDriver::default(),
                },
            )?;
            Ok(LinalgObservedOutcome::Lstsq {
                x: result.x,
                residuals: result.residuals,
                rank: result.rank,
                singular_values: result.singular_values,
            })
        }
        LinalgCase::Pinv {
            mode,
            a,
            atol,
            rtol,
            check_finite,
            ..
        } => {
            let result = pinv(
                a,
                PinvOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                    atol: *atol,
                    rtol: *rtol,
                },
            )?;
            Ok(LinalgObservedOutcome::Pinv {
                values: result.pseudo_inverse,
                rank: result.rank,
            })
        }
        LinalgCase::Qr {
            mode,
            a,
            check_finite,
            ..
        } => {
            let result = qr(
                a,
                DecompOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                },
            )?;
            Ok(LinalgObservedOutcome::Matrix { values: result.r })
        }
        LinalgCase::Svd {
            mode,
            a,
            check_finite,
            ..
        } => {
            let result = svd(
                a,
                DecompOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                },
            )?;
            Ok(LinalgObservedOutcome::Vector {
                values: result.s,
                warning_ill_conditioned: false,
            })
        }
        LinalgCase::Cholesky {
            mode,
            a,
            lower,
            check_finite,
            ..
        } => {
            let result = cholesky(
                a,
                lower.unwrap_or(false),
                DecompOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                },
            )?;
            Ok(LinalgObservedOutcome::Matrix {
                values: result.factor,
            })
        }
        LinalgCase::Eig {
            mode,
            a,
            check_finite,
            ..
        } => {
            let result = eig(
                a,
                DecompOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                },
            )?;
            let mut values = Vec::with_capacity(result.eigenvalues_re.len());
            for (re, im) in result.eigenvalues_re.into_iter().zip(result.eigenvalues_im) {
                if im.abs() > 1e-12 {
                    return Err(LinalgError::InvalidArgument {
                        detail: "fixture eig comparison expects real eigenvalues".to_owned(),
                    });
                }
                values.push(re);
            }
            values.sort_by(f64::total_cmp);
            Ok(LinalgObservedOutcome::Vector {
                values,
                warning_ill_conditioned: false,
            })
        }
        LinalgCase::Eigh {
            mode,
            a,
            check_finite,
            ..
        } => {
            let result = eigh(
                a,
                DecompOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                },
            )?;
            Ok(LinalgObservedOutcome::Vector {
                values: result.eigenvalues,
                warning_ill_conditioned: false,
            })
        }
    }
}

fn execute_linalg_case_with_differential_audit(
    case: &LinalgCase,
    audit_ledger: &fsci_linalg::SyncSharedAuditLedger,
) -> Result<LinalgObservedOutcome, LinalgError> {
    match case {
        LinalgCase::Solve {
            mode,
            a,
            b,
            assume_a,
            lower,
            transposed,
            check_finite,
            ..
        } => {
            let mut portfolio = SolverPortfolio::new(*mode, 64);
            let result = solve_with_audit(
                a,
                b,
                SolveOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                    assume_a: assume_a.clone().map(Into::into),
                    lower: lower.unwrap_or(false),
                    transposed: transposed.unwrap_or(false),
                },
                &mut portfolio,
                audit_ledger,
            )?;
            Ok(LinalgObservedOutcome::Vector {
                values: result.x,
                warning_ill_conditioned: result.warning.is_some(),
            })
        }
        LinalgCase::SolveTriangular {
            mode,
            a,
            b,
            trans,
            lower,
            unit_diagonal,
            check_finite,
            ..
        } => {
            let result = solve_triangular_with_audit(
                a,
                b,
                TriangularSolveOptions {
                    mode: *mode,
                    trans: trans
                        .clone()
                        .map(Into::into)
                        .unwrap_or(TriangularTranspose::NoTranspose),
                    lower: lower.unwrap_or(false),
                    unit_diagonal: unit_diagonal.unwrap_or(false),
                    check_finite: check_finite.unwrap_or(true),
                },
                audit_ledger,
            )?;
            Ok(LinalgObservedOutcome::Vector {
                values: result.x,
                warning_ill_conditioned: result.warning.is_some(),
            })
        }
        LinalgCase::SolveBanded {
            mode,
            l_and_u,
            ab,
            b,
            check_finite,
            ..
        } => {
            let result = solve_banded_with_audit(
                (l_and_u[0], l_and_u[1]),
                ab,
                b,
                SolveOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                    ..SolveOptions::default()
                },
                audit_ledger,
            )?;
            Ok(LinalgObservedOutcome::Vector {
                values: result.x,
                warning_ill_conditioned: result.warning.is_some(),
            })
        }
        LinalgCase::Inv {
            mode,
            a,
            assume_a,
            lower,
            check_finite,
            ..
        } => {
            let result = inv_with_audit(
                a,
                InvOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                    assume_a: assume_a.clone().map(Into::into),
                    lower: lower.unwrap_or(false),
                },
                audit_ledger,
            )?;
            Ok(LinalgObservedOutcome::Matrix {
                values: result.inverse,
            })
        }
        LinalgCase::Det {
            mode,
            a,
            check_finite,
            ..
        } => {
            let value = det_with_audit(a, *mode, check_finite.unwrap_or(true), audit_ledger)?;
            Ok(LinalgObservedOutcome::Scalar { value })
        }
        LinalgCase::Lstsq {
            mode,
            a,
            b,
            cond,
            check_finite,
            ..
        } => {
            let result = lstsq_with_audit(
                a,
                b,
                LstsqOptions {
                    mode: *mode,
                    cond: *cond,
                    check_finite: check_finite.unwrap_or(true),
                    driver: LstsqDriver::default(),
                },
                audit_ledger,
            )?;
            Ok(LinalgObservedOutcome::Lstsq {
                x: result.x,
                residuals: result.residuals,
                rank: result.rank,
                singular_values: result.singular_values,
            })
        }
        LinalgCase::Pinv {
            mode,
            a,
            atol,
            rtol,
            check_finite,
            ..
        } => {
            let result = pinv_with_audit(
                a,
                PinvOptions {
                    mode: *mode,
                    check_finite: check_finite.unwrap_or(true),
                    atol: *atol,
                    rtol: *rtol,
                },
                audit_ledger,
            )?;
            Ok(LinalgObservedOutcome::Pinv {
                values: result.pseudo_inverse,
                rank: result.rank,
            })
        }
        LinalgCase::Qr { .. }
        | LinalgCase::Svd { .. }
        | LinalgCase::Cholesky { .. }
        | LinalgCase::Eig { .. }
        | LinalgCase::Eigh { .. } => execute_linalg_case(case),
    }
}

fn execute_optimize_case(case: &OptimizeCase) -> Result<OptimizeObservedOutcome, OptError> {
    execute_optimize_case_inner(case, None)
}

fn execute_optimize_case_with_differential_audit(
    case: &OptimizeCase,
    audit_ledger: &fsci_opt::SyncSharedAuditLedger,
) -> Result<OptimizeObservedOutcome, OptError> {
    execute_optimize_case_inner(case, Some(audit_ledger))
}

fn execute_optimize_case_inner(
    case: &OptimizeCase,
    audit_ledger: Option<&fsci_opt::SyncSharedAuditLedger>,
) -> Result<OptimizeObservedOutcome, OptError> {
    match case {
        OptimizeCase::Minimize {
            mode,
            method,
            objective,
            x0,
            seed,
            hessp,
            tol,
            maxiter,
            maxfev,
            ..
        } => {
            let hessp = if *hessp {
                Some(hessp_for_minimize_objective(objective)?)
            } else {
                None
            };
            let objective = objective.clone();
            let options = MinimizeOptions {
                method: Some(*method),
                tol: *tol,
                maxiter: *maxiter,
                maxfev: *maxfev,
                seed: *seed,
                hessp,
                mode: *mode,
                ..MinimizeOptions::default()
            };
            let result = match audit_ledger {
                Some(ledger) => minimize_with_audit(
                    |x| evaluate_minimize_objective(&objective, x),
                    x0,
                    options,
                    ledger,
                ),
                None => minimize(|x| evaluate_minimize_objective(&objective, x), x0, options),
            }?;
            Ok(OptimizeObservedOutcome::Minimize {
                x: result.x,
                fun: result.fun,
                success: result.success,
                status: result.status,
            })
        }
        OptimizeCase::DifferentialEvolution {
            objective,
            bounds,
            seed,
            tol,
            maxiter,
            popsize,
            constraints,
            ..
        } => {
            let objective = objective.clone();
            let bounds = optimize_bounds(bounds);
            let defaults = DifferentialEvolutionOptions::default();
            let opts = DifferentialEvolutionOptions {
                seed: *seed,
                tol: tol.unwrap_or(defaults.tol),
                maxiter: maxiter.unwrap_or(defaults.maxiter),
                popsize: popsize.unwrap_or(defaults.popsize),
                ..defaults
            };
            let result = if constraints.is_empty() {
                differential_evolution(
                    |x| evaluate_minimize_objective(&objective, x),
                    &bounds,
                    opts,
                )?
            } else {
                let constraints_owned = constraints.clone();
                let violation = move |x: &[f64]| de_constraint_violation(&constraints_owned, x);
                fsci_opt::differential_evolution_constrained(
                    |x| evaluate_minimize_objective(&objective, x),
                    &bounds,
                    violation,
                    opts,
                )?
            };
            Ok(OptimizeObservedOutcome::Minimize {
                x: result.x,
                fun: result.fun,
                success: result.success,
                status: result.status,
            })
        }
        OptimizeCase::Basinhopping {
            objective,
            x0,
            seed,
            maxiter,
            tol,
            ..
        } => {
            let objective = objective.clone();
            let defaults = BasinhoppingOptions::default();
            let result = basinhopping(
                |x| evaluate_minimize_objective(&objective, x),
                x0,
                BasinhoppingOptions {
                    seed: *seed,
                    niter: maxiter.unwrap_or(defaults.niter),
                    minimizer_tol: *tol,
                    ..defaults
                },
            )?;
            Ok(OptimizeObservedOutcome::Minimize {
                x: result.x,
                fun: result.fun,
                success: result.success,
                status: result.status,
            })
        }
        OptimizeCase::DualAnnealing {
            objective,
            bounds,
            seed,
            maxiter,
            ..
        } => {
            let objective = objective.clone();
            let bounds = optimize_bounds(bounds);
            let result = dual_annealing(
                |x| evaluate_minimize_objective(&objective, x),
                &bounds,
                maxiter.unwrap_or(1000),
                seed.unwrap_or(0),
            )?;
            Ok(OptimizeObservedOutcome::Minimize {
                x: result.x,
                fun: result.fun,
                success: result.success,
                status: result.status,
            })
        }
        OptimizeCase::Brute {
            objective,
            ranges,
            ns,
            ..
        } => {
            let objective = objective.clone();
            let ranges = optimize_bounds(ranges);
            let result = brute(
                |x| evaluate_minimize_objective(&objective, x),
                &ranges,
                ns.unwrap_or(20),
            )?;
            Ok(OptimizeObservedOutcome::Minimize {
                x: result.x,
                fun: result.fun,
                success: result.success,
                status: result.status,
            })
        }
        OptimizeCase::Root {
            mode,
            method,
            objective,
            bracket,
            xtol,
            rtol,
            maxiter,
            ..
        } => {
            let objective = objective.clone();
            let options = RootOptions {
                method: Some(*method),
                xtol: xtol.unwrap_or(RootOptions::default().xtol),
                rtol: rtol.unwrap_or(RootOptions::default().rtol),
                maxiter: maxiter.unwrap_or(RootOptions::default().maxiter),
                mode: *mode,
                ..RootOptions::default()
            };
            let result = root_scalar(
                |x| evaluate_root_objective(&objective, x),
                Some((bracket[0], bracket[1])),
                None,
                None,
                options,
            )?;
            Ok(OptimizeObservedOutcome::Root {
                root: result.root,
                converged: result.converged,
                status: result.status,
            })
        }
    }
}

fn optimize_bounds(bounds: &[[f64; 2]]) -> Vec<(f64, f64)> {
    bounds.iter().map(|[lo, hi]| (*lo, *hi)).collect()
}

// br-7470: nonlinear constraint registry. Names map to scalar-valued
// constraint functions g(x); the spec then tests g(x) ∈ [lb[0], ub[0]].
// Multi-output constraints can be added by extending the closure to
// return Vec<f64>; for now we support scalar.
fn nonlinear_constraint_eval(name: &str, x: &[f64]) -> Option<f64> {
    let x0 = x.first().copied().unwrap_or(0.0);
    let x1 = x.get(1).copied().unwrap_or(0.0);
    match name {
        // Unit-disk constraint: x0^2 + x1^2 (typically tested ≤ 1).
        "unit_disk" => Some(x0 * x0 + x1 * x1),
        // Sum of components — allows e.g. sum(x) >= k.
        "sum_components" => Some(x.iter().sum()),
        _ => None,
    }
}

// br-7470: convert a list of DeConstraintSpec into a single scalar
// violation summing squared boundary excess across all constraints.
// Returns 0 for fully feasible points; positive otherwise.
fn de_constraint_violation(constraints: &[DeConstraintSpec], x: &[f64]) -> f64 {
    let mut total = 0.0;
    for c in constraints {
        match c {
            DeConstraintSpec::Linear { a, lb, ub } => {
                for ((row, &lo), &hi) in a.iter().zip(lb.iter()).zip(ub.iter()) {
                    let dot: f64 = row.iter().zip(x.iter()).map(|(&aij, &xj)| aij * xj).sum();
                    let below = (lo - dot).max(0.0);
                    let above = (dot - hi).max(0.0);
                    total += below * below + above * above;
                }
            }
            DeConstraintSpec::Nonlinear { fn_name, lb, ub } => {
                let Some(g) = nonlinear_constraint_eval(fn_name, x) else {
                    // Unknown constraint name → treat as violated to fail loudly.
                    return f64::INFINITY;
                };
                let lo = lb.first().copied().unwrap_or(f64::NEG_INFINITY);
                let hi = ub.first().copied().unwrap_or(f64::INFINITY);
                let below = (lo - g).max(0.0);
                let above = (g - hi).max(0.0);
                total += below * below + above * above;
            }
        }
    }
    total
}

fn evaluate_minimize_objective(objective: &OptimizeMinObjective, x: &[f64]) -> f64 {
    let x0 = x.first().copied().unwrap_or(0.0);
    let x1 = x.get(1).copied().unwrap_or(0.0);
    match objective {
        OptimizeMinObjective::Rosenbrock2 => {
            let a = 1.0 - x0;
            let b = x1 - x0 * x0;
            a * a + 100.0 * b * b
        }
        OptimizeMinObjective::Ackley2 => {
            let s1 = 0.5 * (x0 * x0 + x1 * x1);
            let s2 = 0.5
                * ((2.0 * std::f64::consts::PI * x0).cos()
                    + (2.0 * std::f64::consts::PI * x1).cos());
            -20.0 * (-0.2 * s1.sqrt()).exp() - s2.exp() + std::f64::consts::E + 20.0
        }
        OptimizeMinObjective::Rastrigin2 => {
            20.0 + (x0 * x0 - 10.0 * (2.0 * std::f64::consts::PI * x0).cos())
                + (x1 * x1 - 10.0 * (2.0 * std::f64::consts::PI * x1).cos())
        }
        OptimizeMinObjective::Himmelblau2 => {
            (x0 * x0 + x1 - 11.0).powi(2) + (x0 + x1 * x1 - 7.0).powi(2)
        }
        OptimizeMinObjective::Sin1d => x0.sin(),
        OptimizeMinObjective::Sphere2 => x0 * x0 + x1 * x1,
        OptimizeMinObjective::ShiftedQuadratic => {
            let dx = x0 - 1.0;
            let dy = x1 + 2.0;
            dx * dx + 4.0 * dy * dy
        }
        OptimizeMinObjective::TranslatedQuadratic => {
            let dx = x0 - 4.0;
            let dy = x1 + 3.0;
            dx * dx + 2.0 * dy * dy
        }
        OptimizeMinObjective::ScaledQuadratic => {
            10.0 * evaluate_minimize_objective(&OptimizeMinObjective::ShiftedQuadratic, x)
        }
        OptimizeMinObjective::RotatedQuadratic => {
            let cx = x0 - 1.0;
            let cy = x1 + 2.0;
            let theta = std::f64::consts::FRAC_PI_6;
            let xr = theta.cos() * cx - theta.sin() * cy;
            let yr = theta.sin() * cx + theta.cos() * cy;
            xr * xr + 4.0 * yr * yr
        }
        OptimizeMinObjective::L1Nonsmooth => x0.abs() + x1.abs(),
        OptimizeMinObjective::FlatQuartic => 1.0e-4 * (x0.powi(4) + x1.powi(4)),
        OptimizeMinObjective::NanBranch => {
            if x0 < 0.0 {
                f64::NAN
            } else {
                x0 * x0 + x1 * x1
            }
        }
    }
}

fn hessp_for_minimize_objective(
    objective: &OptimizeMinObjective,
) -> Result<fsci_opt::types::HesspFunc, OptError> {
    match objective {
        OptimizeMinObjective::ShiftedQuadratic => Ok(shifted_quadratic_hessp),
        OptimizeMinObjective::TranslatedQuadratic => Ok(translated_quadratic_hessp),
        OptimizeMinObjective::ScaledQuadratic => Ok(scaled_quadratic_hessp),
        other => Err(OptError::InvalidArgument {
            detail: format!("hessp fixture support is not defined for {other:?}"),
        }),
    }
}

fn diagonal_hessp(p: &[f64], diag: &[f64]) -> Vec<f64> {
    p.iter()
        .enumerate()
        .map(|(idx, value)| value * diag.get(idx).copied().unwrap_or(0.0))
        .collect()
}

fn shifted_quadratic_hessp(_: &[f64], p: &[f64]) -> Vec<f64> {
    diagonal_hessp(p, &[2.0, 8.0])
}

fn translated_quadratic_hessp(_: &[f64], p: &[f64]) -> Vec<f64> {
    diagonal_hessp(p, &[2.0, 2.0])
}

fn scaled_quadratic_hessp(_: &[f64], p: &[f64]) -> Vec<f64> {
    diagonal_hessp(p, &[20.0, 80.0])
}

fn evaluate_root_objective(objective: &OptimizeRootObjective, x: f64) -> f64 {
    match objective {
        OptimizeRootObjective::CubicMinusTwo => x * x * x - 2.0,
        OptimizeRootObjective::CosMinusX => x.cos() - x,
        OptimizeRootObjective::SinMinusHalf => x.sin() - 0.5,
        OptimizeRootObjective::LinearShift03 => x - 0.3,
        OptimizeRootObjective::NanBranch => {
            if x > 0.8 {
                f64::NAN
            } else {
                x - 0.2
            }
        }
        OptimizeRootObjective::StepDiscontinuous => {
            if x < 0.3 {
                -1.0
            } else {
                1.0
            }
        }
    }
}

fn compare_optimize_case_differential(
    expected: &OptimizeExpectedOutcome,
    observed: &Result<OptimizeObservedOutcome, OptError>,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    match (expected, observed) {
        (
            OptimizeExpectedOutcome::MinimizePoint {
                x,
                fun,
                atol,
                rtol,
                require_success,
                contract_ref,
            },
            Ok(OptimizeObservedOutcome::Minimize {
                x: got_x,
                fun: got_fun,
                success,
                ..
            }),
        ) => {
            let tolerance = resolve_contract_tolerance(contract_ref.as_deref(), *atol, *rtol);
            let Some(got_fun) = *got_fun else {
                return (
                    false,
                    "minimize output missing objective value".to_owned(),
                    None,
                    Some(tolerance),
                );
            };
            let x_diff = if got_x.len() == x.len() {
                max_diff_vec(got_x, x)
            } else {
                f64::INFINITY
            };
            let fun_diff = (got_fun - *fun).abs();
            let pass = allclose_vec(got_x, x, tolerance.atol, tolerance.rtol)
                && allclose_scalar(got_fun, *fun, tolerance.atol, tolerance.rtol)
                && (!require_success.unwrap_or(true) || *success);
            let msg = if pass {
                format!(
                    "minimize matched (x_diff={x_diff:.2e}, fun_diff={fun_diff:.2e}, success={success})"
                )
            } else {
                format!(
                    "minimize mismatch: expected_x={x:?}, got_x={got_x:?}, expected_fun={fun:.8e}, got_fun={got_fun:.8e}, success={success}"
                )
            };
            (pass, msg, Some(x_diff.max(fun_diff)), Some(tolerance))
        }
        (
            OptimizeExpectedOutcome::RootValue {
                root,
                atol,
                rtol,
                require_converged,
                contract_ref,
            },
            Ok(OptimizeObservedOutcome::Root {
                root: got,
                converged,
                ..
            }),
        ) => {
            let tolerance = resolve_contract_tolerance(contract_ref.as_deref(), *atol, *rtol);
            let diff = (*got - *root).abs();
            let pass = allclose_scalar(*got, *root, tolerance.atol, tolerance.rtol)
                && (!require_converged.unwrap_or(true) || *converged);
            let msg = if pass {
                format!("root matched (diff={diff:.2e}, converged={converged})")
            } else {
                format!("root mismatch: expected={root:.8e}, got={got:.8e}, converged={converged}")
            };
            (pass, msg, Some(diff), Some(tolerance))
        }
        (
            OptimizeExpectedOutcome::MinimizeStatus { status, success },
            Ok(OptimizeObservedOutcome::Minimize {
                status: got_status,
                success: got_success,
                ..
            }),
        ) => {
            let pass = *status == *got_status && *success == *got_success;
            let msg = if pass {
                "minimize status matched".to_owned()
            } else {
                format!(
                    "minimize status mismatch: expected status={status:?}, success={success}; got status={got_status:?}, success={got_success}"
                )
            };
            (pass, msg, None, None)
        }
        (
            OptimizeExpectedOutcome::RootStatus { status, converged },
            Ok(OptimizeObservedOutcome::Root {
                status: got_status,
                converged: got_converged,
                ..
            }),
        ) => {
            let pass = *status == *got_status && *converged == *got_converged;
            let msg = if pass {
                "root status matched".to_owned()
            } else {
                format!(
                    "root status mismatch: expected status={status:?}, converged={converged}; got status={got_status:?}, converged={got_converged}"
                )
            };
            (pass, msg, None, None)
        }
        (OptimizeExpectedOutcome::Error { error }, Err(actual)) => {
            let pass = matches_error_contract(&actual.to_string(), error);
            let msg = if pass {
                "error matched".to_owned()
            } else {
                format!("error mismatch: expected=`{error}`, got=`{actual}`")
            };
            (pass, msg, None, None)
        }
        (expected, observed) => (
            false,
            format!("shape mismatch: expected {expected:?}, got {observed:?}"),
            None,
            None,
        ),
    }
}

fn fixture_interpolate_kind_to_runtime(kind: InterpolateInterpKind) -> FsciInterpKind {
    match kind {
        InterpolateInterpKind::Linear => FsciInterpKind::Linear,
        InterpolateInterpKind::Nearest => FsciInterpKind::Nearest,
    }
}

fn fixture_regular_grid_method_to_runtime(
    method: InterpolateRegularGridMethod,
) -> FsciRegularGridMethod {
    match method {
        InterpolateRegularGridMethod::Linear => FsciRegularGridMethod::Linear,
        InterpolateRegularGridMethod::Nearest => FsciRegularGridMethod::Nearest,
        InterpolateRegularGridMethod::Pchip => FsciRegularGridMethod::Pchip,
        InterpolateRegularGridMethod::Cubic => FsciRegularGridMethod::Cubic,
        InterpolateRegularGridMethod::Quintic => FsciRegularGridMethod::Quintic,
    }
}

fn finite_spline_bc_value(name: &str, value: f64) -> Result<f64, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!(
            "{name} boundary condition derivative must be finite"
        ))
    }
}

fn fixture_spline_bc_to_runtime(bc: InterpolateSplineBc) -> Result<FsciSplineBc, String> {
    match bc {
        InterpolateSplineBc::Natural => Ok(FsciSplineBc::Natural),
        InterpolateSplineBc::NotAKnot => Ok(FsciSplineBc::NotAKnot),
        InterpolateSplineBc::Clamped {
            left_derivative,
            right_derivative,
        } => Ok(FsciSplineBc::Clamped(
            finite_spline_bc_value("left", left_derivative)?,
            finite_spline_bc_value("right", right_derivative)?,
        )),
        InterpolateSplineBc::Periodic => Ok(FsciSplineBc::Periodic),
        InterpolateSplineBc::Tuple {
            left_order,
            left_value,
            right_order,
            right_value,
        } => {
            if left_order == 1 && right_order == 1 {
                Ok(FsciSplineBc::Clamped(
                    finite_spline_bc_value("left", left_value)?,
                    finite_spline_bc_value("right", right_value)?,
                ))
            } else if left_order == 2 && right_order == 2 && left_value == 0.0 && right_value == 0.0
            {
                Ok(FsciSplineBc::Natural)
            } else {
                Err("only first-derivative tuple BCs and zero second-derivative natural tuple BCs are supported".to_owned())
            }
        }
    }
}

fn execute_interpolate_case(case: &InterpolateCase) -> InterpolateObservedOutcome {
    match case {
        InterpolateCase::Interp1d {
            mode,
            kind,
            x,
            y,
            x_new,
            bounds_error,
            fill_value,
            ..
        } => {
            let options = Interp1dOptions {
                kind: fixture_interpolate_kind_to_runtime(*kind),
                mode: *mode,
                bounds_error: bounds_error.unwrap_or(true),
                fill_value: *fill_value,
                ..Interp1dOptions::default()
            };
            match Interp1d::new(x, y, options) {
                Ok(interpolator) => match interpolator.eval_many(x_new) {
                    Ok(values) => InterpolateObservedOutcome::Vector(values),
                    Err(error) => InterpolateObservedOutcome::Error(error.to_string()),
                },
                Err(error) => InterpolateObservedOutcome::Error(error.to_string()),
            }
        }
        InterpolateCase::RegularGridInterpolator {
            method,
            points,
            values,
            xi,
            bounds_error,
            fill_value,
            ..
        } => match RegularGridInterpolator::new(
            points.clone(),
            values.clone(),
            fixture_regular_grid_method_to_runtime(*method),
            bounds_error.unwrap_or(true),
            *fill_value,
        ) {
            Ok(interpolator) => match interpolator.eval_many(xi) {
                Ok(values) => InterpolateObservedOutcome::Vector(values),
                Err(error) => InterpolateObservedOutcome::Error(error.to_string()),
            },
            Err(error) => InterpolateObservedOutcome::Error(error.to_string()),
        },
        InterpolateCase::CubicSpline {
            x, y, x_new, bc, ..
        } => {
            let runtime_bc =
                match fixture_spline_bc_to_runtime(bc.unwrap_or(InterpolateSplineBc::Natural)) {
                    Ok(runtime_bc) => runtime_bc,
                    Err(error) => return InterpolateObservedOutcome::Error(error),
                };
            match CubicSplineStandalone::new(x, y, runtime_bc) {
                Ok(spline) => InterpolateObservedOutcome::Vector(spline.eval_many(x_new)),
                Err(error) => InterpolateObservedOutcome::Error(error.to_string()),
            }
        }
        InterpolateCase::BSpline {
            knots,
            coefficients,
            degree,
            x_new,
            ..
        } => match BSpline::new(knots.clone(), coefficients.clone(), *degree) {
            Ok(spline) => InterpolateObservedOutcome::Vector(spline.eval_many(x_new)),
            Err(error) => InterpolateObservedOutcome::Error(error.to_string()),
        },
    }
}

fn compare_interpolate_case_differential(
    case: &InterpolateCase,
    observed: &InterpolateObservedOutcome,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    match (case.expected(), observed) {
        (
            InterpolateExpectedOutcome::Vector { values, atol, rtol },
            InterpolateObservedOutcome::Vector(got),
        ) => {
            let tolerance = ToleranceUsed {
                atol: atol.unwrap_or(1.0e-12),
                rtol: rtol.unwrap_or(1.0e-12),
                comparison_mode: "allclose".to_owned(),
            };
            let max_diff = if got.len() == values.len() {
                max_diff_vec(got, values)
            } else {
                f64::INFINITY
            };
            let pass = allclose_vec(got, values, tolerance.atol, tolerance.rtol);
            let msg = if pass {
                format!("interpolate vector matched (max_diff={max_diff:.2e})")
            } else {
                format!(
                    "interpolate vector mismatch: expected={values:?}, got={got:?}, atol={:.2e}, rtol={:.2e}",
                    tolerance.atol, tolerance.rtol
                )
            };
            (pass, msg, Some(max_diff), Some(tolerance))
        }
        (InterpolateExpectedOutcome::Error { error }, InterpolateObservedOutcome::Error(got)) => {
            let pass = matches_error_contract(got, error);
            let msg = if pass {
                "error matched".to_owned()
            } else {
                format!("error mismatch: expected=`{error}`, got=`{got}`")
            };
            (pass, msg, None, None)
        }
        (InterpolateExpectedOutcome::Error { error }, InterpolateObservedOutcome::Vector(got)) => (
            false,
            format!("expected error `{error}` but got vector {got:?}"),
            None,
            None,
        ),
        (InterpolateExpectedOutcome::Vector { .. }, InterpolateObservedOutcome::Error(error)) => (
            false,
            format!("unexpected interpolate error: {error}"),
            None,
            None,
        ),
    }
}

fn fixture_ndimage_boundary_to_runtime(mode: NdimageBoundaryMode) -> FsciNdimageBoundaryMode {
    match mode {
        NdimageBoundaryMode::Reflect => FsciNdimageBoundaryMode::Reflect,
        NdimageBoundaryMode::Constant => FsciNdimageBoundaryMode::Constant,
        NdimageBoundaryMode::Nearest => FsciNdimageBoundaryMode::Nearest,
        NdimageBoundaryMode::Wrap => FsciNdimageBoundaryMode::Wrap,
    }
}

fn execute_ndimage_case(case: &NdimageCase) -> NdimageObservedOutcome {
    match case {
        NdimageCase::GaussianFilter {
            input,
            shape,
            sigma,
            boundary,
            cval,
            ..
        } => match FsciNdimageArray::new(input.clone(), shape.clone()).and_then(|array| {
            ndimage_gaussian_filter(
                &array,
                *sigma,
                fixture_ndimage_boundary_to_runtime(*boundary),
                *cval,
            )
        }) {
            Ok(array) => NdimageObservedOutcome::Array {
                values: array.data,
                shape: array.shape,
            },
            Err(error) => NdimageObservedOutcome::Error(error.to_string()),
        },
        NdimageCase::Label { input, shape, .. } => {
            match FsciNdimageArray::new(input.clone(), shape.clone()).and_then(|array| {
                let (labels, num_features) = ndimage_label(&array)?;
                Ok((labels, num_features))
            }) {
                Ok((labels, num_features)) => NdimageObservedOutcome::Label {
                    labels: labels.data,
                    shape: labels.shape,
                    num_features,
                },
                Err(error) => NdimageObservedOutcome::Error(error.to_string()),
            }
        }
        NdimageCase::BinaryErosion {
            input,
            shape,
            structure_size,
            iterations,
            ..
        } => match FsciNdimageArray::new(input.clone(), shape.clone())
            .and_then(|array| ndimage_binary_erosion(&array, *structure_size, *iterations))
        {
            Ok(array) => NdimageObservedOutcome::Array {
                values: array.data,
                shape: array.shape,
            },
            Err(error) => NdimageObservedOutcome::Error(error.to_string()),
        },
        NdimageCase::BinaryDilation {
            input,
            shape,
            structure_size,
            iterations,
            ..
        } => match FsciNdimageArray::new(input.clone(), shape.clone())
            .and_then(|array| ndimage_binary_dilation(&array, *structure_size, *iterations))
        {
            Ok(array) => NdimageObservedOutcome::Array {
                values: array.data,
                shape: array.shape,
            },
            Err(error) => NdimageObservedOutcome::Error(error.to_string()),
        },
        NdimageCase::DistanceTransformEdt {
            input,
            shape,
            sampling,
            ..
        } => match FsciNdimageArray::new(input.clone(), shape.clone())
            .and_then(|array| ndimage_distance_transform_edt(&array, sampling.as_deref()))
        {
            Ok(array) => NdimageObservedOutcome::Array {
                values: array.data,
                shape: array.shape,
            },
            Err(error) => NdimageObservedOutcome::Error(error.to_string()),
        },
    }
}

fn compare_ndimage_case_differential(
    case: &NdimageCase,
    observed: &NdimageObservedOutcome,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    match (case.expected(), observed) {
        (
            NdimageExpectedOutcome::Array {
                values,
                shape,
                atol,
                rtol,
            },
            NdimageObservedOutcome::Array {
                values: got,
                shape: got_shape,
            },
        ) => {
            let tolerance = ToleranceUsed {
                atol: atol.unwrap_or(1.0e-12),
                rtol: rtol.unwrap_or(1.0e-12),
                comparison_mode: "allclose".to_owned(),
            };
            if got_shape != shape {
                return (
                    false,
                    format!("ndimage shape mismatch: expected={shape:?}, got={got_shape:?}"),
                    Some(f64::INFINITY),
                    Some(tolerance),
                );
            }
            let max_diff = if got.len() == values.len() {
                max_diff_vec(got, values)
            } else {
                f64::INFINITY
            };
            let pass = allclose_vec(got, values, tolerance.atol, tolerance.rtol);
            let msg = if pass {
                format!("ndimage array matched (max_diff={max_diff:.2e})")
            } else {
                format!(
                    "ndimage array mismatch: expected={values:?}, got={got:?}, atol={:.2e}, rtol={:.2e}",
                    tolerance.atol, tolerance.rtol
                )
            };
            (pass, msg, Some(max_diff), Some(tolerance))
        }
        (
            NdimageExpectedOutcome::Label {
                labels,
                shape,
                num_features,
            },
            NdimageObservedOutcome::Label {
                labels: got_labels,
                shape: got_shape,
                num_features: got_features,
            },
        ) => {
            let pass = got_shape == shape && got_labels == labels && got_features == num_features;
            let diff = if pass { 0.0 } else { f64::INFINITY };
            let msg = if pass {
                format!("ndimage labels matched ({num_features} features)")
            } else {
                format!(
                    "ndimage label mismatch: expected shape={shape:?}, labels={labels:?}, num_features={num_features}; got shape={got_shape:?}, labels={got_labels:?}, num_features={got_features}"
                )
            };
            (pass, msg, Some(diff), None)
        }
        (NdimageExpectedOutcome::Error { error }, NdimageObservedOutcome::Error(got)) => {
            let pass = matches_error_contract(got, error);
            let msg = if pass {
                "error matched".to_owned()
            } else {
                format!("error mismatch: expected=`{error}`, got=`{got}`")
            };
            (pass, msg, None, None)
        }
        (NdimageExpectedOutcome::Error { error }, observed) => (
            false,
            format!("expected error `{error}` but got {observed:?}"),
            None,
            None,
        ),
        (expected, NdimageObservedOutcome::Error(error)) => (
            false,
            format!("unexpected ndimage error for expected {expected:?}: {error}"),
            None,
            None,
        ),
        (expected, observed) => (
            false,
            format!("ndimage outcome kind mismatch: expected={expected:?}, got={observed:?}"),
            None,
            None,
        ),
    }
}

fn ndimage_expected_tolerance(expected: &NdimageExpectedOutcome) -> (Option<f64>, Option<f64>) {
    match expected {
        NdimageExpectedOutcome::Array { atol, rtol, .. } => (*atol, *rtol),
        NdimageExpectedOutcome::Label { .. } | NdimageExpectedOutcome::Error { .. } => (None, None),
    }
}

fn ndimage_oracle_case_to_expected(
    case: &NdimageCase,
    oracle_case: &OracleCaseOutput,
) -> Result<NdimageExpectedOutcome, String> {
    let (atol, rtol) = ndimage_expected_tolerance(case.expected());

    match oracle_case.result_kind.as_str() {
        "array" => Ok(NdimageExpectedOutcome::Array {
            values: oracle_result_field(&oracle_case.result, "values", case.case_id())?,
            shape: oracle_result_field(&oracle_case.result, "shape", case.case_id())?,
            atol,
            rtol,
        }),
        "label" => Ok(NdimageExpectedOutcome::Label {
            labels: oracle_result_field(&oracle_case.result, "labels", case.case_id())?,
            shape: oracle_result_field(&oracle_case.result, "shape", case.case_id())?,
            num_features: oracle_result_field(&oracle_case.result, "num_features", case.case_id())?,
        }),
        "error" => Ok(NdimageExpectedOutcome::Error {
            error: oracle_result_field(&oracle_case.result, "error", case.case_id())?,
        }),
        other => Err(format!(
            "oracle result for {} has unsupported result_kind `{other}`",
            case.case_id()
        )),
    }
}

fn compare_ndimage_case_against_oracle(
    case: &NdimageCase,
    oracle_case: &OracleCaseOutput,
    observed: &NdimageObservedOutcome,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    if oracle_case.status != "ok" {
        return match observed {
            NdimageObservedOutcome::Error(actual) => (
                false,
                format!(
                    "oracle errored (`{}`) and rust errored (`{actual}`) too; case is unjudgeable",
                    oracle_case
                        .error
                        .as_deref()
                        .unwrap_or("unknown oracle error"),
                ),
                None,
                None,
            ),
            NdimageObservedOutcome::Array { .. } | NdimageObservedOutcome::Label { .. } => (
                false,
                format!(
                    "oracle errored (`{}`) but rust succeeded",
                    oracle_case
                        .error
                        .as_deref()
                        .unwrap_or("unknown oracle error"),
                ),
                None,
                None,
            ),
        };
    }

    match ndimage_oracle_case_to_expected(case, oracle_case) {
        Ok(expected) => {
            let mut oracle_case_fixture = case.clone();
            match &mut oracle_case_fixture {
                NdimageCase::GaussianFilter { expected: slot, .. }
                | NdimageCase::Label { expected: slot, .. }
                | NdimageCase::BinaryErosion { expected: slot, .. }
                | NdimageCase::BinaryDilation { expected: slot, .. }
                | NdimageCase::DistanceTransformEdt { expected: slot, .. } => *slot = expected,
            }
            compare_ndimage_case_differential(&oracle_case_fixture, observed)
        }
        Err(message) => (false, message, None, None),
    }
}

fn interpolate_expected_tolerance(
    expected: &InterpolateExpectedOutcome,
) -> (Option<f64>, Option<f64>) {
    match expected {
        InterpolateExpectedOutcome::Vector { atol, rtol, .. } => (*atol, *rtol),
        InterpolateExpectedOutcome::Error { .. } => (None, None),
    }
}

fn interpolate_oracle_case_to_expected(
    case: &InterpolateCase,
    oracle_case: &OracleCaseOutput,
) -> Result<InterpolateExpectedOutcome, String> {
    let (atol, rtol) = interpolate_expected_tolerance(case.expected());

    match oracle_case.result_kind.as_str() {
        "vector" => Ok(InterpolateExpectedOutcome::Vector {
            values: oracle_result_field(&oracle_case.result, "values", case.case_id())?,
            atol,
            rtol,
        }),
        "error" => Ok(InterpolateExpectedOutcome::Error {
            error: oracle_result_field(&oracle_case.result, "error", case.case_id())?,
        }),
        other => Err(format!(
            "oracle result for {} has unsupported result_kind `{other}`",
            case.case_id()
        )),
    }
}

fn compare_interpolate_case_against_oracle(
    case: &InterpolateCase,
    oracle_case: &OracleCaseOutput,
    observed: &InterpolateObservedOutcome,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    if oracle_case.status != "ok" {
        return match observed {
            InterpolateObservedOutcome::Error(actual) => (
                false,
                format!(
                    "oracle errored (`{}`) and rust errored (`{actual}`) too; case is unjudgeable",
                    oracle_case
                        .error
                        .as_deref()
                        .unwrap_or("unknown oracle error"),
                ),
                None,
                None,
            ),
            InterpolateObservedOutcome::Vector(_) => (
                false,
                format!(
                    "oracle errored (`{}`) but rust succeeded",
                    oracle_case
                        .error
                        .as_deref()
                        .unwrap_or("unknown oracle error"),
                ),
                None,
                None,
            ),
        };
    }

    match interpolate_oracle_case_to_expected(case, oracle_case) {
        Ok(expected) => {
            let mut oracle_case_fixture = case.clone();
            match &mut oracle_case_fixture {
                InterpolateCase::Interp1d { expected: slot, .. }
                | InterpolateCase::RegularGridInterpolator { expected: slot, .. }
                | InterpolateCase::CubicSpline { expected: slot, .. }
                | InterpolateCase::BSpline { expected: slot, .. } => *slot = expected,
            }
            compare_interpolate_case_differential(&oracle_case_fixture, observed)
        }
        Err(message) => (false, message, None, None),
    }
}

fn decode_io_hex_bytes(content_hex: &str) -> Result<Vec<u8>, String> {
    let trimmed = content_hex.trim();
    if !trimmed.len().is_multiple_of(2) {
        return Err(format!("hex byte payload has odd length {}", trimmed.len()));
    }
    let mut bytes = Vec::with_capacity(trimmed.len() / 2);
    for index in (0..trimmed.len()).step_by(2) {
        let byte = u8::from_str_radix(&trimmed[index..index + 2], 16)
            .map_err(|e| format!("parse hex byte at offset {index}: {e}"))?;
        bytes.push(byte);
    }
    Ok(bytes)
}

fn first_mat_array_as_matrix(arrays: Vec<MatArray>) -> Result<(usize, usize, Vec<f64>), String> {
    let first = arrays
        .into_iter()
        .next()
        .ok_or_else(|| "MAT file did not contain any arrays".to_owned())?;
    Ok((first.rows, first.cols, first.data))
}

fn execute_io_case(case: &IoCase) -> IoObservedOutcome {
    match case {
        IoCase::Mmread { content, .. } => match mmread(content) {
            Ok(matrix) => IoObservedOutcome::Matrix {
                rows: matrix.rows,
                cols: matrix.cols,
                values: matrix.data,
            },
            Err(error) => IoObservedOutcome::Error(error.to_string()),
        },
        IoCase::Mmwrite {
            rows, cols, data, ..
        } => match mmwrite(*rows, *cols, data).and_then(|content| mmread(&content)) {
            Ok(matrix) => IoObservedOutcome::Matrix {
                rows: matrix.rows,
                cols: matrix.cols,
                values: matrix.data,
            },
            Err(error) => IoObservedOutcome::Error(error.to_string()),
        },
        IoCase::Loadmat { content_hex, .. } => {
            match decode_io_hex_bytes(content_hex)
                .map_err(|error| error.to_string())
                .and_then(|bytes| loadmat(&bytes).map_err(|error| error.to_string()))
                .and_then(first_mat_array_as_matrix)
            {
                Ok((rows, cols, values)) => IoObservedOutcome::Matrix { rows, cols, values },
                Err(error) => IoObservedOutcome::Error(error),
            }
        }
        IoCase::Savemat {
            name,
            rows,
            cols,
            data,
            ..
        } => {
            let array = MatArray {
                name: name.clone(),
                rows: *rows,
                cols: *cols,
                data: data.clone(),
            };
            match savemat(&[array])
                .and_then(|bytes| loadmat(&bytes))
                .map_err(|error| error.to_string())
                .and_then(first_mat_array_as_matrix)
            {
                Ok((rows, cols, values)) => IoObservedOutcome::Matrix { rows, cols, values },
                Err(error) => IoObservedOutcome::Error(error),
            }
        }
        IoCase::Loadtxt { content, .. } => match loadtxt(content) {
            Ok((rows, cols, values)) => IoObservedOutcome::Matrix { rows, cols, values },
            Err(error) => IoObservedOutcome::Error(error.to_string()),
        },
        IoCase::Savetxt {
            rows,
            cols,
            data,
            delimiter,
            ..
        } => match savetxt(*rows, *cols, data, delimiter).and_then(|content| loadtxt(&content)) {
            Ok((rows, cols, values)) => IoObservedOutcome::Matrix { rows, cols, values },
            Err(error) => IoObservedOutcome::Error(error.to_string()),
        },
        IoCase::WavWrite {
            sample_rate,
            channels,
            data,
            ..
        } => match wav_write(*sample_rate, *channels, data).and_then(|bytes| wav_read(&bytes)) {
            Ok(wav) => IoObservedOutcome::Wav {
                sample_rate: wav.sample_rate,
                channels: wav.channels,
                bits_per_sample: wav.bits_per_sample,
                values: wav.data,
            },
            Err(error) => IoObservedOutcome::Error(error.to_string()),
        },
    }
}

fn compare_io_matrix(
    expected_rows: usize,
    expected_cols: usize,
    expected_values: &[f64],
    atol: Option<f64>,
    rtol: Option<f64>,
    observed: &IoObservedOutcome,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    let tolerance = ToleranceUsed {
        atol: atol.unwrap_or(1.0e-12),
        rtol: rtol.unwrap_or(1.0e-12),
        comparison_mode: "allclose".to_owned(),
    };
    match observed {
        IoObservedOutcome::Matrix { rows, cols, values } => {
            if *rows != expected_rows || *cols != expected_cols {
                return (
                    false,
                    format!(
                        "io matrix shape mismatch: expected={expected_rows}x{expected_cols}, got={rows}x{cols}"
                    ),
                    Some(f64::INFINITY),
                    Some(tolerance),
                );
            }
            let max_diff = if values.len() == expected_values.len() {
                max_diff_vec(values, expected_values)
            } else {
                f64::INFINITY
            };
            let pass = allclose_vec(values, expected_values, tolerance.atol, tolerance.rtol);
            let msg = if pass {
                format!("io matrix matched (max_diff={max_diff:.2e})")
            } else {
                format!(
                    "io matrix mismatch: expected={expected_values:?}, got={values:?}, atol={:.2e}, rtol={:.2e}",
                    tolerance.atol, tolerance.rtol
                )
            };
            (pass, msg, Some(max_diff), Some(tolerance))
        }
        IoObservedOutcome::Error(error) => (
            false,
            format!("unexpected io error for matrix expected output: {error}"),
            None,
            None,
        ),
        other => (
            false,
            format!("io outcome kind mismatch for matrix expected output: got {other:?}"),
            None,
            None,
        ),
    }
}

fn compare_io_wav(
    expected_sample_rate: u32,
    expected_channels: u16,
    expected_bits_per_sample: u16,
    expected_values: &[f64],
    atol: Option<f64>,
    rtol: Option<f64>,
    observed: &IoObservedOutcome,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    let tolerance = ToleranceUsed {
        atol: atol.unwrap_or(1.0e-4),
        rtol: rtol.unwrap_or(1.0e-4),
        comparison_mode: "allclose".to_owned(),
    };
    match observed {
        IoObservedOutcome::Wav {
            sample_rate,
            channels,
            bits_per_sample,
            values,
        } => {
            if *sample_rate != expected_sample_rate
                || *channels != expected_channels
                || *bits_per_sample != expected_bits_per_sample
            {
                return (
                    false,
                    format!(
                        "io wav metadata mismatch: expected rate={expected_sample_rate}, channels={expected_channels}, bits={expected_bits_per_sample}; got rate={sample_rate}, channels={channels}, bits={bits_per_sample}"
                    ),
                    Some(f64::INFINITY),
                    Some(tolerance),
                );
            }
            let max_diff = if values.len() == expected_values.len() {
                max_diff_vec(values, expected_values)
            } else {
                f64::INFINITY
            };
            let pass = allclose_vec(values, expected_values, tolerance.atol, tolerance.rtol);
            let msg = if pass {
                format!("io wav matched (max_diff={max_diff:.2e})")
            } else {
                format!(
                    "io wav mismatch: expected={expected_values:?}, got={values:?}, atol={:.2e}, rtol={:.2e}",
                    tolerance.atol, tolerance.rtol
                )
            };
            (pass, msg, Some(max_diff), Some(tolerance))
        }
        IoObservedOutcome::Error(error) => (
            false,
            format!("unexpected io error for wav expected output: {error}"),
            None,
            None,
        ),
        other => (
            false,
            format!("io outcome kind mismatch for wav expected output: got {other:?}"),
            None,
            None,
        ),
    }
}

fn compare_io_case_differential(
    case: &IoCase,
    observed: &IoObservedOutcome,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    match case.expected() {
        IoExpectedOutcome::Matrix {
            rows,
            cols,
            values,
            atol,
            rtol,
        } => compare_io_matrix(*rows, *cols, values, *atol, *rtol, observed),
        IoExpectedOutcome::Wav {
            sample_rate,
            channels,
            bits_per_sample,
            values,
            atol,
            rtol,
        } => compare_io_wav(
            *sample_rate,
            *channels,
            *bits_per_sample,
            values,
            *atol,
            *rtol,
            observed,
        ),
        IoExpectedOutcome::Error { error } => match observed {
            IoObservedOutcome::Error(got) => {
                let pass = matches_error_contract(got, error);
                let msg = if pass {
                    "error matched".to_owned()
                } else {
                    format!("error mismatch: expected=`{error}`, got=`{got}`")
                };
                (pass, msg, None, None)
            }
            other => (
                false,
                format!("expected io error `{error}` but got {other:?}"),
                None,
                None,
            ),
        },
    }
}

fn io_expected_tolerance(expected: &IoExpectedOutcome) -> (Option<f64>, Option<f64>) {
    match expected {
        IoExpectedOutcome::Matrix { atol, rtol, .. }
        | IoExpectedOutcome::Wav { atol, rtol, .. } => (*atol, *rtol),
        IoExpectedOutcome::Error { .. } => (None, None),
    }
}

fn io_oracle_case_to_expected(
    case: &IoCase,
    oracle_case: &OracleCaseOutput,
) -> Result<IoExpectedOutcome, String> {
    let (atol, rtol) = io_expected_tolerance(case.expected());

    match oracle_case.result_kind.as_str() {
        "matrix" => Ok(IoExpectedOutcome::Matrix {
            rows: oracle_result_field(&oracle_case.result, "rows", case.case_id())?,
            cols: oracle_result_field(&oracle_case.result, "cols", case.case_id())?,
            values: oracle_result_field(&oracle_case.result, "values", case.case_id())?,
            atol,
            rtol,
        }),
        "wav" => Ok(IoExpectedOutcome::Wav {
            sample_rate: oracle_result_field(&oracle_case.result, "sample_rate", case.case_id())?,
            channels: oracle_result_field(&oracle_case.result, "channels", case.case_id())?,
            bits_per_sample: oracle_result_field(
                &oracle_case.result,
                "bits_per_sample",
                case.case_id(),
            )?,
            values: oracle_result_field(&oracle_case.result, "values", case.case_id())?,
            atol,
            rtol,
        }),
        "error" => Ok(IoExpectedOutcome::Error {
            error: oracle_result_field(&oracle_case.result, "error", case.case_id())?,
        }),
        other => Err(format!(
            "oracle result for {} has unsupported result_kind `{other}`",
            case.case_id()
        )),
    }
}

fn compare_io_case_against_oracle(
    case: &IoCase,
    oracle_case: &OracleCaseOutput,
    observed: &IoObservedOutcome,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    if oracle_case.status != "ok" {
        return match observed {
            IoObservedOutcome::Error(actual) => (
                false,
                format!(
                    "oracle errored (`{}`) and rust errored (`{actual}`) too; case is unjudgeable",
                    oracle_case
                        .error
                        .as_deref()
                        .unwrap_or("unknown oracle error"),
                ),
                None,
                None,
            ),
            IoObservedOutcome::Matrix { .. } | IoObservedOutcome::Wav { .. } => (
                false,
                format!(
                    "oracle errored (`{}`) but rust succeeded",
                    oracle_case
                        .error
                        .as_deref()
                        .unwrap_or("unknown oracle error"),
                ),
                None,
                None,
            ),
        };
    }

    match io_oracle_case_to_expected(case, oracle_case) {
        Ok(expected) => {
            let mut oracle_case_fixture = case.clone();
            match &mut oracle_case_fixture {
                IoCase::Mmread { expected: slot, .. }
                | IoCase::Mmwrite { expected: slot, .. }
                | IoCase::Loadmat { expected: slot, .. }
                | IoCase::Savemat { expected: slot, .. }
                | IoCase::Loadtxt { expected: slot, .. }
                | IoCase::Savetxt { expected: slot, .. }
                | IoCase::WavWrite { expected: slot, .. } => *slot = expected,
            }
            compare_io_case_differential(&oracle_case_fixture, observed)
        }
        Err(message) => (false, message, None, None),
    }
}

fn compare_linalg_case(
    expected: &LinalgExpectedOutcome,
    observed: &Result<LinalgObservedOutcome, LinalgError>,
) -> (bool, String) {
    match (expected, observed) {
        (
            LinalgExpectedOutcome::Vector {
                values,
                atol,
                rtol,
                expect_warning_ill_conditioned,
            },
            Ok(LinalgObservedOutcome::Vector {
                values: got,
                warning_ill_conditioned,
            }),
        ) => {
            let vectors_match = allclose_vec(got, values, *atol, *rtol);
            let warning_match = expect_warning_ill_conditioned
                .is_none_or(|expect| expect == *warning_ill_conditioned);
            let pass = vectors_match && warning_match;
            let msg = if pass {
                "linalg vector output matched expected tolerance policy".to_owned()
            } else {
                format!(
                    "vector mismatch: expected={values:?}, got={got:?}, atol={atol}, rtol={rtol}, expected_warning={expect_warning_ill_conditioned:?}, actual_warning={warning_ill_conditioned}"
                )
            };
            (pass, msg)
        }
        (
            LinalgExpectedOutcome::Matrix { values, atol, rtol },
            Ok(LinalgObservedOutcome::Matrix { values: got }),
        ) => {
            let pass = allclose_matrix(got, values, *atol, *rtol);
            let msg = if pass {
                "linalg matrix output matched expected tolerance policy".to_owned()
            } else {
                format!(
                    "matrix mismatch: expected={values:?}, got={got:?}, atol={atol}, rtol={rtol}"
                )
            };
            (pass, msg)
        }
        (
            LinalgExpectedOutcome::Scalar { value, atol, rtol },
            Ok(LinalgObservedOutcome::Scalar { value: got }),
        ) => {
            let pass = allclose_scalar(*got, *value, *atol, *rtol);
            let msg = if pass {
                "linalg scalar output matched expected tolerance policy".to_owned()
            } else {
                format!("scalar mismatch: expected={value}, got={got}, atol={atol}, rtol={rtol}")
            };
            (pass, msg)
        }
        (
            LinalgExpectedOutcome::Lstsq {
                x,
                residuals,
                rank,
                singular_values,
                atol,
                rtol,
            },
            Ok(LinalgObservedOutcome::Lstsq {
                x: got_x,
                residuals: got_res,
                rank: got_rank,
                singular_values: got_s,
            }),
        ) => {
            let pass = *rank == *got_rank
                && allclose_vec(got_x, x, *atol, *rtol)
                && allclose_vec(got_res, residuals, *atol, *rtol)
                && allclose_vec(got_s, singular_values, *atol, *rtol);
            let msg = if pass {
                "linalg lstsq output matched expected tolerance policy".to_owned()
            } else {
                format!(
                    "lstsq mismatch: expected_x={x:?}, got_x={got_x:?}, expected_res={residuals:?}, got_res={got_res:?}, expected_rank={rank}, got_rank={got_rank}, expected_s={singular_values:?}, got_s={got_s:?}"
                )
            };
            (pass, msg)
        }
        (
            LinalgExpectedOutcome::Pinv {
                values,
                rank,
                atol,
                rtol,
            },
            Ok(LinalgObservedOutcome::Pinv {
                values: got,
                rank: got_rank,
            }),
        ) => {
            let pass = *rank == *got_rank && allclose_matrix(got, values, *atol, *rtol);
            let msg = if pass {
                "linalg pinv output matched expected tolerance policy".to_owned()
            } else {
                format!(
                    "pinv mismatch: expected_values={values:?}, got_values={got:?}, expected_rank={rank}, got_rank={got_rank}, atol={atol}, rtol={rtol}"
                )
            };
            (pass, msg)
        }
        (LinalgExpectedOutcome::Error { error }, Err(actual)) => {
            let pass = matches_error_contract(&actual.to_string(), error);
            let msg = if pass {
                "linalg error matched expected contract".to_owned()
            } else {
                format!("error mismatch: expected=`{error}`, got=`{actual}`")
            };
            (pass, msg)
        }
        (expected, got) => (false, format!("expected {expected:?} but observed {got:?}")),
    }
}

fn allclose_scalar(actual: f64, expected: f64, atol: f64, rtol: f64) -> bool {
    if actual.is_nan() && expected.is_nan() {
        return true;
    }
    if actual.is_infinite() || expected.is_infinite() {
        return actual == expected;
    }
    (actual - expected).abs() <= atol + rtol * expected.abs()
}

/// Parse a fixture-supplied error-kind string into the typed
/// `FsciSpecialErrorKind` enum. Returns `None` when the string doesn't
/// correspond to a known variant — the caller then surfaces a judgeable
/// mismatch rather than silently passing. Per frankenscipy-yxml: typed
/// comparison replaces `format!("{:?}", ...)` string-match so the
/// fixture is robust to Debug-repr changes AND surfaces typos as
/// explicit errors.
fn parse_special_error_kind(s: &str) -> Option<FsciSpecialErrorKind> {
    match s {
        "DomainError" => Some(FsciSpecialErrorKind::DomainError),
        "FixtureSchemaError" => Some(FsciSpecialErrorKind::FixtureSchemaError),
        "PoleInput" => Some(FsciSpecialErrorKind::PoleInput),
        "NonFiniteInput" => Some(FsciSpecialErrorKind::NonFiniteInput),
        "CancellationRisk" => Some(FsciSpecialErrorKind::CancellationRisk),
        "OverflowRisk" => Some(FsciSpecialErrorKind::OverflowRisk),
        "SingularityRisk" => Some(FsciSpecialErrorKind::SingularityRisk),
        "NotYetImplemented" => Some(FsciSpecialErrorKind::NotYetImplemented),
        "ShapeMismatch" => Some(FsciSpecialErrorKind::ShapeMismatch),
        _ => None,
    }
}

/// Match a fixture-declared error-contract string against the actual Rust
/// Display output of an error. Uses normalized substring match (case-fold,
/// whitespace collapsed) rather than exact equality, so rewording the error
/// message (e.g. "must be non-negative" → "must be >= 0") doesn't flip a
/// previously-passing conformance case to fail.
///
/// Per /testing-conformance-harnesses anti-pattern 6 ("Check error message
/// strings"): checking error types/categories is preferred, but until all
/// fixtures are migrated to typed error_kind fields, this substring rule
/// keeps the blast radius of harmless rewording contained. See
/// frankenscipy-gnun.
fn matches_error_contract(actual: &str, expected: &str) -> bool {
    let normalize = |s: &str| -> String {
        s.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    };
    let n_actual = normalize(actual);
    let n_expected = normalize(expected);
    n_expected.is_empty() || n_actual == n_expected || n_actual.contains(&n_expected)
}

fn allclose_vec(actual: &[f64], expected: &[f64], atol: f64, rtol: f64) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    actual
        .iter()
        .zip(expected.iter())
        .all(|(a, e)| allclose_scalar(*a, *e, atol, rtol))
}

fn allclose_matrix(actual: &[Vec<f64>], expected: &[Vec<f64>], atol: f64, rtol: f64) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    actual
        .iter()
        .zip(expected.iter())
        .all(|(a_row, e_row)| allclose_vec(a_row, e_row, atol, rtol))
}

fn build_packet_report(
    packet_id: String,
    family: String,
    case_results: Vec<CaseResult>,
) -> PacketReport {
    let passed_cases = case_results.iter().filter(|r| r.passed).count();
    let failed_cases = case_results.len().saturating_sub(passed_cases);
    PacketReport {
        schema_version: packet_report_schema_v1(),
        packet_id,
        family,
        case_results,
        passed_cases,
        failed_cases,
        fixture_path: None,
        oracle_status: None,
        differential_case_results: None,
        // Leave as Unspecified to preserve the serialized-bytes shape for
        // existing hash-pinned consumers (raptorq sidecar drills etc.).
        // Self-check writers that want the explicit tag — including the
        // evidence_p2c*.rs final-pack tests — should overwrite this field
        // to ReportKind::SelfCheck after construction. Per frankenscipy-fytm.
        report_kind: ReportKind::Unspecified,
        generated_unix_ms: now_unix_ms(),
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ContractTolerancePolicy {
    comparison_mode: Option<String>,
    default_atol: Option<f64>,
    default_rtol: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ContractEntry {
    function_name: String,
    tolerance_policy: ContractTolerancePolicy,
}

#[derive(Debug, Clone, Deserialize)]
struct ContractTable {
    #[serde(default)]
    #[cfg_attr(not(test), allow(dead_code))]
    packet_id: Option<String>,
    #[serde(default)]
    #[cfg_attr(not(test), allow(dead_code))]
    domain: Option<String>,
    contracts: Vec<ContractEntry>,
}

static OPTIMIZE_CONTRACT_TABLE: OnceLock<Option<ContractTable>> = OnceLock::new();
static SPECIAL_CONTRACT_TABLE: OnceLock<Option<ContractTable>> = OnceLock::new();
static ARRAY_API_CONTRACT_TABLE: OnceLock<Option<ContractTable>> = OnceLock::new();
static CLUSTER_CONTRACT_TABLE: OnceLock<Option<ContractTable>> = OnceLock::new();
static SPATIAL_CONTRACT_TABLE: OnceLock<Option<ContractTable>> = OnceLock::new();
static SIGNAL_CONTRACT_TABLE: OnceLock<Option<ContractTable>> = OnceLock::new();
static STATS_CONTRACT_TABLE: OnceLock<Option<ContractTable>> = OnceLock::new();
static INTEGRATE_CONTRACT_TABLE: OnceLock<Option<ContractTable>> = OnceLock::new();

fn load_optimize_contract_table() -> Option<&'static ContractTable> {
    OPTIMIZE_CONTRACT_TABLE
        .get_or_init(|| {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/artifacts/P2C-003/contracts/contract_table.json");
            let raw = fs::read_to_string(path).ok()?;
            serde_json::from_str::<ContractTable>(&raw).ok()
        })
        .as_ref()
}

fn load_special_contract_table() -> Option<&'static ContractTable> {
    SPECIAL_CONTRACT_TABLE
        .get_or_init(|| {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/artifacts/P2C-006/contracts/contract_table.json");
            let raw = fs::read_to_string(path).ok()?;
            serde_json::from_str::<ContractTable>(&raw).ok()
        })
        .as_ref()
}

fn load_array_api_contract_table() -> Option<&'static ContractTable> {
    ARRAY_API_CONTRACT_TABLE
        .get_or_init(|| {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/artifacts/P2C-007/contracts/contract_table.json");
            let raw = fs::read_to_string(path).ok()?;
            serde_json::from_str::<ContractTable>(&raw).ok()
        })
        .as_ref()
}

fn load_cluster_contract_table() -> Option<&'static ContractTable> {
    CLUSTER_CONTRACT_TABLE
        .get_or_init(|| {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/artifacts/P2C-009/contracts/contract_table.json");
            let raw = fs::read_to_string(path).ok()?;
            serde_json::from_str::<ContractTable>(&raw).ok()
        })
        .as_ref()
}

fn load_spatial_contract_table() -> Option<&'static ContractTable> {
    SPATIAL_CONTRACT_TABLE
        .get_or_init(|| {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/artifacts/P2C-010/contracts/contract_table.json");
            let raw = fs::read_to_string(path).ok()?;
            serde_json::from_str::<ContractTable>(&raw).ok()
        })
        .as_ref()
}

fn load_signal_contract_table() -> Option<&'static ContractTable> {
    SIGNAL_CONTRACT_TABLE
        .get_or_init(|| {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/artifacts/P2C-011/contracts/contract_table.json");
            let raw = fs::read_to_string(path).ok()?;
            serde_json::from_str::<ContractTable>(&raw).ok()
        })
        .as_ref()
}

fn load_stats_contract_table() -> Option<&'static ContractTable> {
    STATS_CONTRACT_TABLE
        .get_or_init(|| {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/artifacts/P2C-012/contracts/contract_table.json");
            let raw = fs::read_to_string(path).ok()?;
            serde_json::from_str::<ContractTable>(&raw).ok()
        })
        .as_ref()
}

fn load_integrate_contract_table() -> Option<&'static ContractTable> {
    INTEGRATE_CONTRACT_TABLE
        .get_or_init(|| {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/artifacts/P2C-013/contracts/contract_table.json");
            let raw = fs::read_to_string(path).ok()?;
            serde_json::from_str::<ContractTable>(&raw).ok()
        })
        .as_ref()
}

fn resolve_table_contract_tolerance(
    table: Option<&ContractTable>,
    contract_ref: Option<&str>,
    explicit_atol: Option<f64>,
    explicit_rtol: Option<f64>,
    default_atol: f64,
    default_rtol: f64,
    default_comparison_mode: &str,
) -> ToleranceUsed {
    if let (Some(atol), Some(rtol)) = (explicit_atol, explicit_rtol) {
        return ToleranceUsed {
            atol,
            rtol,
            comparison_mode: default_comparison_mode.to_owned(),
        };
    }

    if let Some(contract_ref) = contract_ref.filter(|reference| !reference.is_empty())
        && let Some(table) = table
        && let Some(entry) = table
            .contracts
            .iter()
            .find(|candidate| candidate.function_name == contract_ref)
    {
        return ToleranceUsed {
            atol: explicit_atol
                .or(entry.tolerance_policy.default_atol)
                .unwrap_or(default_atol),
            rtol: explicit_rtol
                .or(entry.tolerance_policy.default_rtol)
                .unwrap_or(default_rtol),
            comparison_mode: entry
                .tolerance_policy
                .comparison_mode
                .clone()
                .unwrap_or_else(|| default_comparison_mode.to_owned()),
        };
    }

    ToleranceUsed {
        atol: explicit_atol.unwrap_or(default_atol),
        rtol: explicit_rtol.unwrap_or(default_rtol),
        comparison_mode: default_comparison_mode.to_owned(),
    }
}

fn resolve_contract_tolerance(
    contract_ref: Option<&str>,
    explicit_atol: Option<f64>,
    explicit_rtol: Option<f64>,
) -> ToleranceUsed {
    if let (Some(atol), Some(rtol)) = (explicit_atol, explicit_rtol) {
        return ToleranceUsed {
            atol,
            rtol,
            comparison_mode: "mixed".to_owned(),
        };
    }

    if let Some(contract_ref) = contract_ref
        && let Some(table) = load_optimize_contract_table()
        && let Some(entry) = table
            .contracts
            .iter()
            .find(|candidate| candidate.function_name == contract_ref)
    {
        return ToleranceUsed {
            atol: explicit_atol
                .or(entry.tolerance_policy.default_atol)
                .unwrap_or(1.0e-9),
            rtol: explicit_rtol
                .or(entry.tolerance_policy.default_rtol)
                .unwrap_or(1.0e-6),
            comparison_mode: entry
                .tolerance_policy
                .comparison_mode
                .clone()
                .unwrap_or_else(|| "mixed".to_owned()),
        };
    }

    ToleranceUsed {
        atol: explicit_atol.unwrap_or(1.0e-9),
        rtol: explicit_rtol.unwrap_or(1.0e-6),
        comparison_mode: "mixed".to_owned(),
    }
}

fn resolve_special_contract_tolerance(
    contract_ref: Option<&str>,
    explicit_atol: Option<f64>,
    explicit_rtol: Option<f64>,
) -> ToleranceUsed {
    if let (Some(atol), Some(rtol)) = (explicit_atol, explicit_rtol) {
        return ToleranceUsed {
            atol,
            rtol,
            comparison_mode: "mixed".to_owned(),
        };
    }

    if let Some(contract_ref) = contract_ref
        && let Some(table) = load_special_contract_table()
        && let Some(entry) = table
            .contracts
            .iter()
            .find(|candidate| candidate.function_name == contract_ref)
    {
        return ToleranceUsed {
            atol: explicit_atol
                .or(entry.tolerance_policy.default_atol)
                .unwrap_or(1.0e-9),
            rtol: explicit_rtol
                .or(entry.tolerance_policy.default_rtol)
                .unwrap_or(1.0e-6),
            comparison_mode: entry
                .tolerance_policy
                .comparison_mode
                .clone()
                .unwrap_or_else(|| "mixed".to_owned()),
        };
    }

    ToleranceUsed {
        atol: explicit_atol.unwrap_or(1.0e-9),
        rtol: explicit_rtol.unwrap_or(1.0e-6),
        comparison_mode: "mixed".to_owned(),
    }
}

fn resolve_array_api_contract_tolerance(
    contract_ref: Option<&str>,
    explicit_atol: Option<f64>,
    explicit_rtol: Option<f64>,
) -> ToleranceUsed {
    if let (Some(atol), Some(rtol)) = (explicit_atol, explicit_rtol) {
        return ToleranceUsed {
            atol,
            rtol,
            comparison_mode: "mixed".to_owned(),
        };
    }

    if let Some(contract_ref) = contract_ref
        && let Some(table) = load_array_api_contract_table()
        && let Some(entry) = table
            .contracts
            .iter()
            .find(|candidate| candidate.function_name == contract_ref)
    {
        let comparison_mode = entry
            .tolerance_policy
            .comparison_mode
            .clone()
            .unwrap_or_else(|| "exact".to_owned());
        return ToleranceUsed {
            atol: explicit_atol
                .or(entry.tolerance_policy.default_atol)
                .unwrap_or(0.0),
            rtol: explicit_rtol
                .or(entry.tolerance_policy.default_rtol)
                .unwrap_or(0.0),
            comparison_mode,
        };
    }

    ToleranceUsed {
        atol: explicit_atol.unwrap_or(0.0),
        rtol: explicit_rtol.unwrap_or(0.0),
        comparison_mode: "exact".to_owned(),
    }
}

fn resolve_integrate_contract_tolerance(
    contract_ref: Option<&str>,
    explicit_atol: Option<f64>,
    explicit_rtol: Option<f64>,
) -> ToleranceUsed {
    resolve_table_contract_tolerance(
        load_integrate_contract_table(),
        contract_ref,
        explicit_atol,
        explicit_rtol,
        1.0e-12,
        1.0e-12,
        "allclose",
    )
}

fn resolve_stats_contract_tolerance(
    contract_ref: Option<&str>,
    explicit_atol: Option<f64>,
    explicit_rtol: Option<f64>,
) -> ToleranceUsed {
    resolve_table_contract_tolerance(
        load_stats_contract_table(),
        contract_ref,
        explicit_atol,
        explicit_rtol,
        1.0e-10,
        1.0e-10,
        "allclose",
    )
}

fn resolve_signal_contract_tolerance(
    contract_ref: Option<&str>,
    explicit_atol: Option<f64>,
    explicit_rtol: Option<f64>,
) -> ToleranceUsed {
    resolve_table_contract_tolerance(
        load_signal_contract_table(),
        contract_ref,
        explicit_atol,
        explicit_rtol,
        1.0e-10,
        1.0e-10,
        "allclose",
    )
}

fn resolve_spatial_contract_tolerance(
    contract_ref: Option<&str>,
    explicit_atol: Option<f64>,
    explicit_rtol: Option<f64>,
) -> ToleranceUsed {
    resolve_table_contract_tolerance(
        load_spatial_contract_table(),
        contract_ref,
        explicit_atol,
        explicit_rtol,
        1.0e-10,
        1.0e-10,
        "allclose",
    )
}

fn resolve_cluster_contract_tolerance(
    contract_ref: Option<&str>,
    explicit_atol: Option<f64>,
    explicit_rtol: Option<f64>,
) -> ToleranceUsed {
    resolve_table_contract_tolerance(
        load_cluster_contract_table(),
        contract_ref,
        explicit_atol,
        explicit_rtol,
        1.0e-10,
        1.0e-10,
        "allclose",
    )
}

pub fn generate_raptorq_sidecar(payload: &[u8]) -> Result<RaptorQSidecar, HarnessError> {
    let symbol_size = 128usize;
    let source_symbols = chunk_payload(payload, symbol_size);
    let k = source_symbols.len();
    let repair_symbols = (k / 5).max(1);
    let seed = hash(payload).as_bytes()[0] as u64 + 1337;

    let encoder = SystematicEncoder::new(&source_symbols, symbol_size, seed).ok_or_else(|| {
        HarnessError::RaptorQ("systematic encoder initialization failed".to_owned())
    })?;

    let mut repair_hashes = Vec::with_capacity(repair_symbols);
    let mut repair_payloads = Vec::with_capacity(repair_symbols);
    for esi in k as u32..(k as u32 + repair_symbols as u32) {
        let symbol = encoder.repair_symbol(esi);
        repair_hashes.push(hash(&symbol).to_hex().to_string());
        repair_payloads.push(hex_encode(&symbol));
    }

    Ok(RaptorQSidecar {
        schema_version: 1,
        source_hash: hash(payload).to_hex().to_string(),
        symbol_size,
        seed,
        source_symbols: k,
        repair_symbols,
        repair_symbol_hashes: repair_hashes,
        repair_symbol_payloads_hex: repair_payloads,
    })
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

pub fn chunk_payload(payload: &[u8], symbol_size: usize) -> Vec<Vec<u8>> {
    if payload.is_empty() {
        return vec![vec![0u8; symbol_size]];
    }

    let mut chunks = Vec::with_capacity(payload.len().div_ceil(symbol_size));
    for chunk in payload.chunks(symbol_size) {
        let mut symbol = vec![0u8; symbol_size];
        let len = chunk.len();
        symbol[..len].copy_from_slice(chunk);
        chunks.push(symbol);
    }
    chunks
}

fn as_os_str(input: &str) -> &OsStr {
    OsStr::new(input)
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

// ═══════════════════════════════════════════════════════════════════
// Differential Conformance Harness (bd-3jh.4)
// Generic API: run_differential_test(fixture_path, oracle_config) -> ConformanceReport
// ═══════════════════════════════════════════════════════════════════

/// Oracle configuration for differential conformance testing.
#[derive(Debug, Clone)]
pub struct DifferentialOracleConfig {
    /// Path to Python interpreter (e.g., ".venv-py314/bin/python3").
    pub python_path: PathBuf,
    /// Path to the oracle script that runs SciPy and emits JSON.
    pub script_path: PathBuf,
    /// Maximum time in seconds for oracle execution.
    pub timeout_secs: u64,
    /// If true, oracle failures are hard errors. If false, oracle_missing is acceptable.
    pub required: bool,
}

impl Default for DifferentialOracleConfig {
    fn default() -> Self {
        // Sentinel script path that does not exist; callers MUST route
        // via `resolve_differential_oracle_config(family)` to pick the
        // correct scipy_<family>_oracle.py, or set `script_path` explicitly.
        // See bpmm — the prior default pointed at scipy_linalg_oracle.py,
        // which silently misrouted non-linalg families in any code path
        // that skipped the resolve step.
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            python_path: PathBuf::from("python3"),
            script_path: manifest.join("python_oracle/NO_DEFAULT_ORACLE_SET.py"),
            timeout_secs: 30,
            required: false,
        }
    }
}

/// Status of the Python/SciPy oracle for a given test run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum OracleStatus {
    /// Oracle ran successfully and produced results.
    Available,
    /// Oracle was not available (SciPy not installed, script missing, etc.).
    Missing { reason: String },
    /// Oracle exceeded the configured timeout.
    TimedOut,
    /// Oracle ran but produced an error.
    Failed { reason: String },
    /// Oracle was not invoked (not configured or not needed).
    Skipped,
}

/// Tolerance values that were used for a particular case comparison.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToleranceUsed {
    pub atol: f64,
    pub rtol: f64,
    /// Descriptive tag naming the comparator that the case-specific code
    /// actually ran. **Does NOT branch comparator behaviour** — every
    /// comparator today uses allclose semantics via
    /// `allclose_scalar(atol, rtol)` or an equivalent inline formula. The
    /// tag is emitted into the ToleranceUsed record purely as a telemetry
    /// / audit hint for dashboards and post-hoc analysis so reviewers can
    /// see at a glance which comparator class was invoked. If/when a
    /// comparator dispatch actually varies on this field, the behaviour
    /// change should be called out explicitly in the bead log. Per
    /// frankenscipy-ka5i (currently the field is set at 12+ sites but
    /// never read by any branching code).
    pub comparison_mode: String,
}

/// Per-case result from differential conformance testing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DifferentialCaseResult {
    /// Unique case identifier within the fixture.
    pub case_id: String,
    /// Whether the case passed the conformance check.
    pub passed: bool,
    /// Human-readable description of the result or failure.
    pub message: String,
    /// Maximum absolute difference between Rust and expected output.
    /// None if the case involves error matching rather than numeric comparison.
    pub max_diff: Option<f64>,
    /// The tolerance values used for this comparison.
    pub tolerance_used: Option<ToleranceUsed>,
    /// Status of the oracle for this particular case.
    pub oracle_status: OracleStatus,
}

/// Conformance report produced by `run_differential_test`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConformanceReport {
    /// Path to the fixture file that was tested.
    pub fixture_path: String,
    /// Packet identifier extracted from the fixture.
    pub packet_id: String,
    /// Family name from the fixture (e.g., "validate_tol", "linalg_core").
    pub family: String,
    /// Number of cases that passed.
    pub pass_count: usize,
    /// Number of cases that failed.
    pub fail_count: usize,
    /// Overall oracle status for the test run.
    pub oracle_status: OracleStatus,
    /// Per-case results with max_diff and tolerance information.
    pub per_case_results: Vec<DifferentialCaseResult>,
    /// Timestamp when this report was generated.
    pub generated_unix_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DriftCaseDiff {
    pub case_id: String,
    pub passed: bool,
    pub message: String,
    pub max_diff: Option<f64>,
    pub tolerance_used: Option<ToleranceUsed>,
    pub oracle_status: OracleStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DriftDiffReport {
    pub schema_version: u8,
    pub packet_id: String,
    pub family: String,
    pub fixture_path: String,
    pub generated_unix_ms: u128,
    pub oracle_status: OracleStatus,
    pub total_cases: usize,
    pub drifted_cases: usize,
    pub cases: Vec<DriftCaseDiff>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LiveOracleCaptureReport {
    pub schema_version: u8,
    pub generated_unix_ms: u128,
    pub fixture_root: String,
    pub total_packets: usize,
    pub total_cases: usize,
    pub drifted_cases: usize,
    pub oracle_available_packets: usize,
    pub oracle_unavailable_packets: usize,
    pub packets: Vec<DriftDiffReport>,
}

pub const DEFAULT_MAX_DRIFTED_CASES_PER_PACKET: usize = 0;
pub const MIN_ZERO_DRIFT_GATE_ORACLE_AVAILABLE_PACKETS: usize = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DriftGateThreshold {
    pub packet_id: String,
    pub max_drifted_cases: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DriftGatePacketResult {
    pub packet_id: String,
    pub family: String,
    pub drifted_cases: usize,
    pub max_allowed_drifted_cases: usize,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DriftGateReport {
    pub schema_version: u8,
    pub generated_unix_ms: u128,
    pub total_packets: usize,
    pub total_cases: usize,
    pub total_drifted_cases: usize,
    pub failed_packets: usize,
    pub packets: Vec<DriftGatePacketResult>,
}

impl DriftGateReport {
    #[must_use]
    pub const fn passed(&self) -> bool {
        self.failed_packets == 0
    }
}

#[must_use]
pub fn default_zero_drift_thresholds() -> Vec<DriftGateThreshold> {
    PacketFamily::ALL
        .iter()
        .map(|family| DriftGateThreshold {
            packet_id: family.packet_id().to_owned(),
            max_drifted_cases: DEFAULT_MAX_DRIFTED_CASES_PER_PACKET,
        })
        .collect()
}

#[must_use]
pub fn evaluate_drift_gate(
    report: &LiveOracleCaptureReport,
    thresholds: &[DriftGateThreshold],
) -> DriftGateReport {
    let packets = report
        .packets
        .iter()
        .map(|packet| {
            let max_allowed_drifted_cases = thresholds
                .iter()
                .find(|threshold| threshold.packet_id == packet.packet_id)
                .map_or(DEFAULT_MAX_DRIFTED_CASES_PER_PACKET, |threshold| {
                    threshold.max_drifted_cases
                });
            DriftGatePacketResult {
                packet_id: packet.packet_id.clone(),
                family: packet.family.clone(),
                drifted_cases: packet.drifted_cases,
                max_allowed_drifted_cases,
                passed: packet.drifted_cases <= max_allowed_drifted_cases,
            }
        })
        .collect::<Vec<_>>();
    let failed_packets = packets.iter().filter(|packet| !packet.passed).count();

    DriftGateReport {
        schema_version: 1,
        generated_unix_ms: report.generated_unix_ms,
        total_packets: report.total_packets,
        total_cases: report.total_cases,
        total_drifted_cases: report.drifted_cases,
        failed_packets,
        packets,
    }
}

pub fn validate_zero_drift_oracle_coverage(report: &LiveOracleCaptureReport) -> Result<(), String> {
    if report.total_packets == 0 {
        return Err(
            "zero-drift gate found no differential packets; refusing to pass an empty oracle gate"
                .to_string(),
        );
    }
    if report.oracle_available_packets < MIN_ZERO_DRIFT_GATE_ORACLE_AVAILABLE_PACKETS {
        return Err(format!(
            "zero-drift gate had {} oracle-backed packet(s), below required minimum {}; \
             {} packet(s) were oracle-unavailable",
            report.oracle_available_packets,
            MIN_ZERO_DRIFT_GATE_ORACLE_AVAILABLE_PACKETS,
            report.oracle_unavailable_packets
        ));
    }
    Ok(())
}

impl From<&DifferentialCaseResult> for CaseResult {
    fn from(result: &DifferentialCaseResult) -> Self {
        Self {
            case_id: result.case_id.clone(),
            passed: result.passed,
            message: result.message.clone(),
        }
    }
}

impl From<&ConformanceReport> for PacketReport {
    fn from(report: &ConformanceReport) -> Self {
        Self {
            schema_version: packet_report_schema_v2(),
            packet_id: report.packet_id.clone(),
            family: report.family.clone(),
            case_results: report.per_case_results.iter().map(Into::into).collect(),
            passed_cases: report.pass_count,
            failed_cases: report.fail_count,
            fixture_path: Some(report.fixture_path.clone()),
            oracle_status: Some(report.oracle_status.clone()),
            differential_case_results: Some(report.per_case_results.clone()),
            // ConformanceReport comes from the differential lane; mark as
            // OracleBacked when the oracle_status indicates availability,
            // otherwise SelfCheck. Per frankenscipy-fytm.
            report_kind: if matches!(report.oracle_status, OracleStatus::Available) {
                ReportKind::OracleBacked
            } else {
                ReportKind::SelfCheck
            },
            generated_unix_ms: report.generated_unix_ms,
        }
    }
}

/// Fixture envelope: minimal structure to detect fixture type.
#[derive(Debug, Clone, Deserialize)]
struct FixtureEnvelope {
    family: String,
}

fn differential_artifact_dir_for_fixture(fixture_path: &Path, packet_id: &str) -> PathBuf {
    fixture_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("artifacts")
        .join(packet_id)
        .join("differential")
}

pub(crate) fn differential_audit_ledger_path_for_fixture(
    fixture_path: &Path,
    packet_id: &str,
) -> PathBuf {
    differential_artifact_dir_for_fixture(fixture_path, packet_id).join("audit_ledger.jsonl")
}

/// Neutral shared audit ledger for differential runners that don't wire
/// through a family-specific ledger (fsci-array_api, fsci-special,
/// fsci-optimize, fsci-casp, fsci-cluster, fsci-spatial, fsci-signal,
/// fsci-stats, fsci-integrate-differential). Per frankenscipy-mhuk: each
/// differential runner now emits an audit ledger artifact — possibly
/// empty — so downstream consumers can distinguish "family ran, no audit
/// events" from "family was never audit-instrumented."
fn neutral_audit_ledger() -> std::sync::Arc<std::sync::Mutex<AuditLedger>> {
    std::sync::Arc::new(std::sync::Mutex::new(AuditLedger::new()))
}

fn emit_differential_audit_ledger_for_fixture(
    fixture_path: &Path,
    packet_id: &str,
    ledger: &AuditLedger,
) -> Result<PathBuf, HarnessError> {
    let output_dir = differential_artifact_dir_for_fixture(fixture_path, packet_id);
    crate::e2e::emit_audit_ledger(&output_dir, ledger).map_err(|error| match error {
        crate::e2e::E2eOrchestratorError::LogWrite { path, source } => {
            HarnessError::ArtifactIo { path, source }
        }
        crate::e2e::E2eOrchestratorError::LogSerialize(source) => {
            HarnessError::RaptorQ(source.to_string())
        }
        other => HarnessError::RaptorQ(other.to_string()),
    })
}

fn panic_payload_message(payload: Box<dyn Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(message) => (*message).to_owned(),
            Err(_) => "non-string panic payload".to_owned(),
        },
    }
}

fn run_case_with_panic_capture<F>(
    case_id: &str,
    oracle_status: &OracleStatus,
    audit_ledger: Option<&std::sync::Mutex<AuditLedger>>,
    run: F,
) -> DifferentialCaseResult
where
    F: FnOnce() -> (bool, String, Option<f64>, Option<ToleranceUsed>),
{
    match panic::catch_unwind(AssertUnwindSafe(run)) {
        Ok((passed, message, max_diff, tolerance_used)) => DifferentialCaseResult {
            case_id: case_id.to_owned(),
            passed,
            message,
            max_diff,
            tolerance_used,
            oracle_status: oracle_status.clone(),
        },
        Err(payload) => {
            if let Some(ledger) = audit_ledger {
                ledger.clear_poison();
            }
            DifferentialCaseResult {
                case_id: case_id.to_owned(),
                passed: false,
                message: format!("PANIC: {}", panic_payload_message(payload)),
                max_diff: None,
                tolerance_used: None,
                oracle_status: oracle_status.clone(),
            }
        }
    }
}

fn recover_sync_audit_ledger<'a>(
    ledger: &'a std::sync::Mutex<AuditLedger>,
) -> std::sync::MutexGuard<'a, AuditLedger> {
    match ledger.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            ledger.clear_poison();
            poisoned.into_inner()
        }
    }
}

/// Run a differential conformance test against a fixture file.
///
/// This is the generic entry point for the conformance harness. It:
/// 1. Loads and parses the fixture file
/// 2. Detects the fixture type (validate_tol, linalg_core, etc.)
/// 3. Runs the Rust implementation against each case
/// 4. Optionally runs the Python/SciPy oracle for comparison
/// 5. Computes max_diff between Rust and expected outputs per case
///
/// If the oracle is unavailable and `oracle_config.required == false`,
/// the report marks oracle_status as `Missing` and still validates
/// against the fixture's embedded expected values.
///
/// # Oracle wiring state
///
/// Step 4 currently captures Python oracle output for linalg packet
/// helpers and for the generic differential lanes that explicitly call
/// `capture_python_oracle_inner` (stats, array_api, constants). Other
/// family-specific scripts may exist on disk and be routable through
/// `default_differential_oracle_script_path(family)`, but a runner that
/// only probes availability and then compares Rust output against the
/// fixture's embedded expected values remains **self-checking**, not
/// oracle-backed.
pub fn run_differential_test(
    fixture_path: &Path,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let raw = fs::read_to_string(fixture_path).map_err(|source| HarnessError::FixtureIo {
        path: fixture_path.to_path_buf(),
        source,
    })?;

    let envelope: FixtureEnvelope =
        serde_json::from_str(&raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let family = envelope.family.as_str();
    let resolved_oracle_config = resolve_differential_oracle_config(oracle_config, family);
    // Exact-family dispatch per frankenscipy-ubkg. The previous cascade
    // used `family.contains(...)` which caused fragile matches (e.g.
    // family='integrate.validate_tol' matched BOTH 'validate_tol' AND
    // 'integrate'; family='sparse_ops' would never match 'sparse' because
    // there was no sparse arm, but the unambiguous-substring anti-pattern
    // is still wrong even where it happens to produce the intended result
    // today). Match on exact canonical family names.
    let report = match family {
        // validate_tol packet (also integrate.validate_tol for the
        // shared-tolerance lane).
        "validate_tol" | "integrate.validate_tol" => {
            run_differential_validate_tol(fixture_path, &raw, &resolved_oracle_config)
        }
        "linalg_core" | "linalg" => {
            run_differential_linalg(fixture_path, &raw, &resolved_oracle_config)
        }
        "array_api" | "array_api_core" => {
            run_differential_array_api(fixture_path, &raw, &resolved_oracle_config)
        }
        "special_core" | "special" => {
            run_differential_special(fixture_path, &raw, &resolved_oracle_config)
        }
        "fft_core" | "fft" => run_differential_fft(fixture_path, &raw, &resolved_oracle_config),
        "optimize_core" | "optimize" => {
            run_differential_optimize(fixture_path, &raw, &resolved_oracle_config)
        }
        "interpolate_core" | "interpolate" => {
            run_differential_interpolate(fixture_path, &raw, &resolved_oracle_config)
        }
        "io_core" | "io" => run_differential_io(fixture_path, &raw, &resolved_oracle_config),
        "ndimage_core" | "ndimage" => {
            run_differential_ndimage(fixture_path, &raw, &resolved_oracle_config)
        }
        "runtime_casp" | "casp" | "casp_core" => {
            run_differential_casp(fixture_path, &raw, &resolved_oracle_config)
        }
        "cluster_core" | "cluster" => {
            run_differential_cluster(fixture_path, &raw, &resolved_oracle_config)
        }
        "spatial_core" | "spatial" => {
            run_differential_spatial(fixture_path, &raw, &resolved_oracle_config)
        }
        "signal_core" | "signal" => {
            run_differential_signal(fixture_path, &raw, &resolved_oracle_config)
        }
        "stats_core" | "stats" => {
            run_differential_stats(fixture_path, &raw, &resolved_oracle_config)
        }
        "integrate_core" | "integrate" => {
            run_differential_integrate(fixture_path, &raw, &resolved_oracle_config)
        }
        "constants_core" | "constants" => {
            run_differential_constants(fixture_path, &raw, &resolved_oracle_config)
        }
        _ => Err(HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source: serde::de::Error::custom(format!("unknown fixture family: {family}")),
        }),
    }?;

    let _ = write_differential_parity_artifacts(fixture_path, &report)?;
    Ok(report)
}

fn run_differential_validate_tol(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: PacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let oracle_status = probe_oracle_availability(oracle_config);
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());
    let audit_ledger = integrate_sync_audit_ledger();

    for case in &fixture.cases {
        per_case_results.push(run_case_with_panic_capture(
            &case.case_id,
            &oracle_status,
            Some(audit_ledger.as_ref()),
            || {
                let outcome = validate_tol_with_audit(
                    case.rtol.clone().into(),
                    case.atol.clone().into(),
                    case.n,
                    case.mode,
                    Some(&audit_ledger),
                );

                match (&case.expected, outcome) {
                    (
                        ExpectedOutcome::Ok {
                            rtol,
                            atol,
                            warning_rtol_clamped,
                        },
                        Ok(actual),
                    ) => {
                        let actual_warning = !actual.warnings.is_empty();
                        let expected_rtol: ToleranceValue = rtol.clone().into();
                        let expected_atol: ToleranceValue = atol.clone().into();
                        let pass = actual.rtol == expected_rtol
                            && actual.atol == expected_atol
                            && actual_warning == *warning_rtol_clamped;
                        let msg = if pass {
                            "tolerance contract matched".to_owned()
                        } else {
                            format!(
                                "tolerance mismatch: expected rtol={expected_rtol:?}, atol={expected_atol:?}, warning={warning_rtol_clamped}; got rtol={:?}, atol={:?}, warning={actual_warning}",
                                actual.rtol, actual.atol
                            )
                        };
                        (pass, msg, None, None)
                    }
                    (ExpectedOutcome::Error { error }, Err(actual)) => {
                        let pass = matches_error_contract(&actual.to_string(), error);
                        let msg = if pass {
                            "error matched".to_owned()
                        } else {
                            format!("error mismatch: expected `{error}`, got `{actual}`")
                        };
                        (pass, msg, None, None)
                    }
                    (expected, result) => (
                        false,
                        format!("shape mismatch: expected {expected:?}, got {result:?}"),
                        None,
                        None,
                    ),
                }
            },
        ));
    }

    let pass_count = per_case_results.iter().filter(|r| r.passed).count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);
    {
        let ledger = recover_sync_audit_ledger(audit_ledger.as_ref());
        let _ =
            emit_differential_audit_ledger_for_fixture(fixture_path, &fixture.packet_id, &ledger)?;
    }

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

fn run_differential_linalg(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: LinalgPacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let oracle_status = probe_oracle_availability(oracle_config);
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());
    let audit_ledger = linalg_sync_audit_ledger();

    for case in &fixture.cases {
        per_case_results.push(run_case_with_panic_capture(
            case.case_id(),
            &oracle_status,
            Some(audit_ledger.as_ref()),
            || {
                let observed = execute_linalg_case_with_differential_audit(case, &audit_ledger);
                compare_linalg_case_differential(case.expected(), &observed)
            },
        ));
    }

    let pass_count = per_case_results.iter().filter(|r| r.passed).count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);
    {
        let ledger = recover_sync_audit_ledger(audit_ledger.as_ref());
        let _ =
            emit_differential_audit_ledger_for_fixture(fixture_path, &fixture.packet_id, &ledger)?;
    }

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

fn run_differential_array_api(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: ArrayApiPacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let resolved_oracle_config = resolve_differential_oracle_config(oracle_config, &fixture.family);
    let probed_oracle_status = probe_oracle_availability(&resolved_oracle_config);
    let mut capture_failure_status = None;
    let oracle_capture = if resolved_oracle_config.required
        || matches!(probed_oracle_status, OracleStatus::Available)
    {
        match capture_python_oracle_inner(
            fixture_path,
            raw,
            &fixture.packet_id,
            &HarnessConfig::default_paths().oracle_root,
            &resolved_oracle_config,
        ) {
            Ok(capture) => Some(capture),
            Err(error) => {
                if resolved_oracle_config.required {
                    return Err(error);
                }
                capture_failure_status = Some(oracle_status_from_capture_error(&error));
                None
            }
        }
    } else {
        None
    };
    let oracle_cases = oracle_capture.as_ref().map(|capture| {
        capture
            .case_outputs
            .iter()
            .map(|case| (case.case_id.as_str(), case))
            .collect::<std::collections::HashMap<_, _>>()
    });
    let oracle_status = match (&oracle_capture, resolved_oracle_config.required) {
        (Some(_), _) => OracleStatus::Available,
        (None, true) => probed_oracle_status.clone(),
        (None, false) => capture_failure_status.unwrap_or(probed_oracle_status),
    };
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());
    let audit_ledger = array_api_sync_audit_ledger();

    for case in &fixture.cases {
        per_case_results.push(run_case_with_panic_capture(
            case.case_id(),
            &oracle_status,
            Some(audit_ledger.as_ref()),
            || {
                let observed = execute_array_api_case_with_differential_audit(case, &audit_ledger);
                match oracle_cases.as_ref() {
                    Some(cases) => match cases.get(case.case_id()) {
                        Some(oracle_case) => {
                            compare_array_api_case_against_oracle(case, oracle_case, &observed)
                        }
                        None => (
                            false,
                            format!("oracle capture missing array_api case `{}`", case.case_id()),
                            None,
                            None,
                        ),
                    },
                    None => compare_array_api_case_differential(case, &observed),
                }
            },
        ));
    }

    let pass_count = per_case_results.iter().filter(|r| r.passed).count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);

    {
        let ledger = recover_sync_audit_ledger(audit_ledger.as_ref());
        let _ =
            emit_differential_audit_ledger_for_fixture(fixture_path, &fixture.packet_id, &ledger)?;
    }

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

fn run_differential_optimize(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: OptimizePacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let oracle_status = probe_oracle_availability(oracle_config);
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());
    let audit_ledger = opt_sync_audit_ledger();

    for case in &fixture.cases {
        per_case_results.push(run_case_with_panic_capture(
            case.case_id(),
            &oracle_status,
            Some(audit_ledger.as_ref()),
            || {
                let observed = execute_optimize_case_with_differential_audit(case, &audit_ledger);
                compare_optimize_case_differential(case.expected(), &observed)
            },
        ));
    }

    let pass_count = per_case_results.iter().filter(|r| r.passed).count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);

    {
        let ledger = recover_sync_audit_ledger(audit_ledger.as_ref());
        let _ =
            emit_differential_audit_ledger_for_fixture(fixture_path, &fixture.packet_id, &ledger)?;
    }

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

fn run_differential_interpolate(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: InterpolatePacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let resolved_oracle_config = resolve_differential_oracle_config(oracle_config, &fixture.family);
    let probed_oracle_status = probe_oracle_availability(&resolved_oracle_config);
    let mut capture_failure_status = None;
    let oracle_capture = if resolved_oracle_config.required
        || matches!(probed_oracle_status, OracleStatus::Available)
    {
        match capture_python_oracle_inner(
            fixture_path,
            raw,
            &fixture.packet_id,
            &HarnessConfig::default_paths().oracle_root,
            &resolved_oracle_config,
        ) {
            Ok(capture) => Some(capture),
            Err(error) => {
                if resolved_oracle_config.required {
                    return Err(error);
                }
                capture_failure_status = Some(oracle_status_from_capture_error(&error));
                None
            }
        }
    } else {
        None
    };
    let oracle_cases = oracle_capture.as_ref().map(|capture| {
        capture
            .case_outputs
            .iter()
            .map(|case| (case.case_id.as_str(), case))
            .collect::<std::collections::HashMap<_, _>>()
    });
    let oracle_status = match (&oracle_capture, resolved_oracle_config.required) {
        (Some(_), _) => OracleStatus::Available,
        (None, true) => probed_oracle_status.clone(),
        (None, false) => capture_failure_status.unwrap_or(probed_oracle_status),
    };
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());
    let audit_ledger = neutral_audit_ledger();

    for case in &fixture.cases {
        per_case_results.push(run_case_with_panic_capture(
            case.case_id(),
            &oracle_status,
            Some(audit_ledger.as_ref()),
            || {
                let observed = execute_interpolate_case(case);
                match oracle_cases.as_ref() {
                    Some(cases) => match cases.get(case.case_id()) {
                        Some(oracle_case) => {
                            compare_interpolate_case_against_oracle(case, oracle_case, &observed)
                        }
                        None => (
                            false,
                            format!(
                                "oracle capture missing interpolate case `{}`",
                                case.case_id()
                            ),
                            None,
                            None,
                        ),
                    },
                    None => compare_interpolate_case_differential(case, &observed),
                }
            },
        ));
    }

    let pass_count = per_case_results.iter().filter(|r| r.passed).count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);

    {
        let ledger = recover_sync_audit_ledger(audit_ledger.as_ref());
        let _ =
            emit_differential_audit_ledger_for_fixture(fixture_path, &fixture.packet_id, &ledger)?;
    }

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

fn run_differential_io(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: IoPacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let resolved_oracle_config = resolve_differential_oracle_config(oracle_config, &fixture.family);
    let probed_oracle_status = probe_oracle_availability(&resolved_oracle_config);
    let mut capture_failure_status = None;
    let oracle_capture = if resolved_oracle_config.required
        || matches!(probed_oracle_status, OracleStatus::Available)
    {
        match capture_python_oracle_inner(
            fixture_path,
            raw,
            &fixture.packet_id,
            &HarnessConfig::default_paths().oracle_root,
            &resolved_oracle_config,
        ) {
            Ok(capture) => Some(capture),
            Err(error) => {
                if resolved_oracle_config.required {
                    return Err(error);
                }
                capture_failure_status = Some(oracle_status_from_capture_error(&error));
                None
            }
        }
    } else {
        None
    };
    let oracle_cases = oracle_capture.as_ref().map(|capture| {
        capture
            .case_outputs
            .iter()
            .map(|case| (case.case_id.as_str(), case))
            .collect::<std::collections::HashMap<_, _>>()
    });
    let oracle_status = match (&oracle_capture, resolved_oracle_config.required) {
        (Some(_), _) => OracleStatus::Available,
        (None, true) => probed_oracle_status.clone(),
        (None, false) => capture_failure_status.unwrap_or(probed_oracle_status),
    };
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());
    let audit_ledger = neutral_audit_ledger();

    for case in &fixture.cases {
        per_case_results.push(run_case_with_panic_capture(
            case.case_id(),
            &oracle_status,
            Some(audit_ledger.as_ref()),
            || {
                let observed = execute_io_case(case);
                match oracle_cases.as_ref() {
                    Some(cases) => match cases.get(case.case_id()) {
                        Some(oracle_case) => {
                            compare_io_case_against_oracle(case, oracle_case, &observed)
                        }
                        None => (
                            false,
                            format!("oracle capture missing io case `{}`", case.case_id()),
                            None,
                            None,
                        ),
                    },
                    None => compare_io_case_differential(case, &observed),
                }
            },
        ));
    }

    let pass_count = per_case_results.iter().filter(|r| r.passed).count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);

    {
        let ledger = recover_sync_audit_ledger(audit_ledger.as_ref());
        let _ =
            emit_differential_audit_ledger_for_fixture(fixture_path, &fixture.packet_id, &ledger)?;
    }

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

fn run_differential_ndimage(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: NdimagePacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let resolved_oracle_config = resolve_differential_oracle_config(oracle_config, &fixture.family);
    let probed_oracle_status = probe_oracle_availability(&resolved_oracle_config);
    let mut capture_failure_status = None;
    let oracle_capture = if resolved_oracle_config.required
        || matches!(probed_oracle_status, OracleStatus::Available)
    {
        match capture_python_oracle_inner(
            fixture_path,
            raw,
            &fixture.packet_id,
            &HarnessConfig::default_paths().oracle_root,
            &resolved_oracle_config,
        ) {
            Ok(capture) => Some(capture),
            Err(error) => {
                if resolved_oracle_config.required {
                    return Err(error);
                }
                capture_failure_status = Some(oracle_status_from_capture_error(&error));
                None
            }
        }
    } else {
        None
    };
    let oracle_cases = oracle_capture.as_ref().map(|capture| {
        capture
            .case_outputs
            .iter()
            .map(|case| (case.case_id.as_str(), case))
            .collect::<std::collections::HashMap<_, _>>()
    });
    let oracle_status = match (&oracle_capture, resolved_oracle_config.required) {
        (Some(_), _) => OracleStatus::Available,
        (None, true) => probed_oracle_status.clone(),
        (None, false) => capture_failure_status.unwrap_or(probed_oracle_status),
    };
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());
    let audit_ledger = neutral_audit_ledger();

    for case in &fixture.cases {
        per_case_results.push(run_case_with_panic_capture(
            case.case_id(),
            &oracle_status,
            Some(audit_ledger.as_ref()),
            || {
                let observed = execute_ndimage_case(case);
                match oracle_cases.as_ref() {
                    Some(cases) => match cases.get(case.case_id()) {
                        Some(oracle_case) => {
                            compare_ndimage_case_against_oracle(case, oracle_case, &observed)
                        }
                        None => (
                            false,
                            format!("oracle capture missing ndimage case `{}`", case.case_id()),
                            None,
                            None,
                        ),
                    },
                    None => compare_ndimage_case_differential(case, &observed),
                }
            },
        ));
    }

    let pass_count = per_case_results.iter().filter(|r| r.passed).count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);

    {
        let ledger = recover_sync_audit_ledger(audit_ledger.as_ref());
        let _ =
            emit_differential_audit_ledger_for_fixture(fixture_path, &fixture.packet_id, &ledger)?;
    }

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

fn run_differential_special(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: SpecialPacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let oracle_status = probe_oracle_availability(oracle_config);
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());
    let audit_ledger = neutral_audit_ledger();

    for case in &fixture.cases {
        per_case_results.push(run_case_with_panic_capture(
            case.case_id(),
            &oracle_status,
            Some(audit_ledger.as_ref()),
            || {
                let observed = execute_special_case(case);
                compare_special_case_differential(case, &observed)
            },
        ));
    }

    let pass_count = per_case_results.iter().filter(|r| r.passed).count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);

    {
        let ledger = recover_sync_audit_ledger(audit_ledger.as_ref());
        let _ =
            emit_differential_audit_ledger_for_fixture(fixture_path, &fixture.packet_id, &ledger)?;
    }

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

fn run_differential_integrate(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: IntegratePacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let oracle_status = probe_oracle_availability(oracle_config);
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());

    for case in &fixture.cases {
        per_case_results.push(run_case_with_panic_capture(
            case.case_id(),
            &oracle_status,
            None,
            || {
                let observed = execute_integrate_case(case);
                compare_integrate_case_differential(case, &observed)
            },
        ));
    }

    let pass_count = per_case_results.iter().filter(|r| r.passed).count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

fn run_differential_stats(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: StatsPacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let resolved_oracle_config = resolve_differential_oracle_config(oracle_config, &fixture.family);
    let probed_oracle_status = probe_oracle_availability(&resolved_oracle_config);
    let mut capture_failure_status = None;
    let oracle_capture = if resolved_oracle_config.required
        || matches!(probed_oracle_status, OracleStatus::Available)
    {
        match capture_python_oracle_inner(
            fixture_path,
            raw,
            &fixture.packet_id,
            &HarnessConfig::default_paths().oracle_root,
            &resolved_oracle_config,
        ) {
            Ok(capture) => Some(capture),
            Err(error) => {
                if resolved_oracle_config.required {
                    return Err(error);
                }
                capture_failure_status = Some(oracle_status_from_capture_error(&error));
                None
            }
        }
    } else {
        None
    };
    let oracle_cases = oracle_capture.as_ref().map(|capture| {
        capture
            .case_outputs
            .iter()
            .map(|case| (case.case_id.as_str(), case))
            .collect::<std::collections::HashMap<_, _>>()
    });
    let oracle_status = match (&oracle_capture, resolved_oracle_config.required) {
        (Some(_), _) => OracleStatus::Available,
        (None, true) => probed_oracle_status.clone(),
        (None, false) => capture_failure_status.unwrap_or(probed_oracle_status),
    };
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());
    let audit_ledger = neutral_audit_ledger();

    for case in &fixture.cases {
        per_case_results.push(run_case_with_panic_capture(
            case.case_id(),
            &oracle_status,
            Some(audit_ledger.as_ref()),
            || {
                let observed = execute_stats_case(case);
                match oracle_cases.as_ref() {
                    Some(cases) => match cases.get(case.case_id()) {
                        Some(oracle_case) => {
                            compare_stats_case_against_oracle(case, oracle_case, &observed)
                        }
                        None => (
                            false,
                            format!("oracle capture missing stats case `{}`", case.case_id()),
                            None,
                            None,
                        ),
                    },
                    None => compare_stats_case_differential(case, &observed),
                }
            },
        ));
    }

    let pass_count = per_case_results.iter().filter(|r| r.passed).count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);

    {
        let ledger = recover_sync_audit_ledger(audit_ledger.as_ref());
        let _ =
            emit_differential_audit_ledger_for_fixture(fixture_path, &fixture.packet_id, &ledger)?;
    }

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

fn run_differential_signal(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: SignalPacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let oracle_status = probe_oracle_availability(oracle_config);
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());
    let audit_ledger = neutral_audit_ledger();

    for case in &fixture.cases {
        per_case_results.push(run_case_with_panic_capture(
            case.case_id(),
            &oracle_status,
            Some(audit_ledger.as_ref()),
            || {
                let observed = execute_signal_case(case);
                compare_signal_case_differential(case, &observed)
            },
        ));
    }

    let pass_count = per_case_results.iter().filter(|r| r.passed).count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);

    {
        let ledger = recover_sync_audit_ledger(audit_ledger.as_ref());
        let _ =
            emit_differential_audit_ledger_for_fixture(fixture_path, &fixture.packet_id, &ledger)?;
    }

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

fn run_differential_spatial(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: SpatialPacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let oracle_status = probe_oracle_availability(oracle_config);
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());
    let audit_ledger = neutral_audit_ledger();

    for case in &fixture.cases {
        per_case_results.push(run_case_with_panic_capture(
            case.case_id(),
            &oracle_status,
            Some(audit_ledger.as_ref()),
            || {
                let observed = execute_spatial_case(case);
                compare_spatial_case_differential(case, &observed)
            },
        ));
    }

    let pass_count = per_case_results
        .iter()
        .filter(|result| result.passed)
        .count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);

    {
        let ledger = recover_sync_audit_ledger(audit_ledger.as_ref());
        let _ =
            emit_differential_audit_ledger_for_fixture(fixture_path, &fixture.packet_id, &ledger)?;
    }

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

fn run_differential_cluster(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: ClusterPacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let oracle_status = probe_oracle_availability(oracle_config);
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());
    let audit_ledger = neutral_audit_ledger();

    for case in &fixture.cases {
        per_case_results.push(run_case_with_panic_capture(
            case.case_id(),
            &oracle_status,
            Some(audit_ledger.as_ref()),
            || {
                let observed = execute_cluster_case(case);
                compare_cluster_case_differential(case, &observed)
            },
        ));
    }

    let pass_count = per_case_results
        .iter()
        .filter(|result| result.passed)
        .count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);

    {
        let ledger = recover_sync_audit_ledger(audit_ledger.as_ref());
        let _ =
            emit_differential_audit_ledger_for_fixture(fixture_path, &fixture.packet_id, &ledger)?;
    }

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

fn run_differential_casp(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: CaspPacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let oracle_status = probe_oracle_availability(oracle_config);
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());
    let audit_ledger = neutral_audit_ledger();

    for case in &fixture.cases {
        per_case_results.push(run_case_with_panic_capture(
            case.case_id(),
            &oracle_status,
            Some(audit_ledger.as_ref()),
            || {
                let observed = execute_casp_case(case);
                compare_casp_case_differential(case, &observed)
            },
        ));
    }

    let pass_count = per_case_results
        .iter()
        .filter(|result| result.passed)
        .count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);

    {
        let ledger = recover_sync_audit_ledger(audit_ledger.as_ref());
        let _ =
            emit_differential_audit_ledger_for_fixture(fixture_path, &fixture.packet_id, &ledger)?;
    }

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

fn run_differential_fft(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: FftPacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let oracle_status = probe_oracle_availability(oracle_config);
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());
    let audit_ledger = fft_sync_audit_ledger();

    for case in &fixture.cases {
        per_case_results.push(run_case_with_panic_capture(
            case.case_id(),
            &oracle_status,
            Some(audit_ledger.as_ref()),
            || {
                let observed = execute_fft_case_with_differential_audit(case, &audit_ledger);
                compare_fft_case_differential(case, &observed)
            },
        ));
    }

    let pass_count = per_case_results
        .iter()
        .filter(|result| result.passed)
        .count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);

    let ledger = recover_sync_audit_ledger(audit_ledger.as_ref());
    if !ledger.is_empty() {
        emit_differential_audit_ledger_for_fixture(fixture_path, &fixture.packet_id, &ledger)?;
    }

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

// ============================================================================
// Constants (fsci-constants / scipy.constants) differential dispatch.
//
// Per frankenscipy-utus: the original FSCI-P2C-016 fixture used a prose-only
// {fixture_id, expected_properties} schema that no dispatch arm could consume.
// This runner uses the canonical {case_id, function, args, expected} shape and
// compares fsci-constants values/helpers against scipy.constants via the
// paired scipy_constants_oracle.py.
// ============================================================================

/// Fixture for constants conformance testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstantsPacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<ConstantsCase>,
}

/// A single constants conformance test case.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstantsCase {
    pub case_id: String,
    pub category: String,
    pub mode: RuntimeMode,
    pub function: String,
    pub args: Vec<serde_json::Value>,
    pub expected: ConstantsExpected,
}

impl ConstantsCase {
    fn case_id(&self) -> &str {
        &self.case_id
    }
}

/// Expected outcome for a constants conformance case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantsExpected {
    pub kind: String,
    #[serde(default)]
    pub value: Option<f64>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub atol: Option<f64>,
    #[serde(default)]
    pub rtol: Option<f64>,
    #[serde(default)]
    pub contract_ref: Option<String>,
}

#[derive(Debug)]
enum ConstantsObserved {
    Scalar(f64),
    Error(String),
}

fn constant_value_by_name(name: &str) -> Option<f64> {
    // Maps Rust const identifiers (case-insensitive) to their compiled
    // values. Kept tight to the public API actually exported by
    // fsci-constants so typos in fixture args fail closed.
    match name.to_ascii_uppercase().as_str() {
        "PI" => Some(fsci_constants::PI),
        "TAU" => Some(fsci_constants::TAU),
        "E" => Some(fsci_constants::E),
        "GOLDEN_RATIO" => Some(fsci_constants::GOLDEN_RATIO),
        "SPEED_OF_LIGHT" | "C" => Some(fsci_constants::SPEED_OF_LIGHT),
        "PLANCK" | "H" => Some(fsci_constants::PLANCK),
        "HBAR" => Some(fsci_constants::HBAR),
        "GRAVITATIONAL_CONSTANT" | "G" => Some(fsci_constants::GRAVITATIONAL_CONSTANT),
        "G_N" => Some(fsci_constants::G_N),
        "ELEMENTARY_CHARGE" | "E_CHARGE" => Some(fsci_constants::ELEMENTARY_CHARGE),
        "GAS_CONSTANT" | "R" => Some(fsci_constants::GAS_CONSTANT),
        "AVOGADRO" | "N_A" => Some(fsci_constants::AVOGADRO),
        "BOLTZMANN" | "K_B" => Some(fsci_constants::BOLTZMANN),
        "STEFAN_BOLTZMANN" | "SIGMA" => Some(fsci_constants::STEFAN_BOLTZMANN),
        "WIEN" => Some(fsci_constants::WIEN),
        "RYDBERG" => Some(fsci_constants::RYDBERG),
        "ELECTRON_MASS" | "M_E" => Some(fsci_constants::ELECTRON_MASS),
        "PROTON_MASS" | "M_P" => Some(fsci_constants::PROTON_MASS),
        "NEUTRON_MASS" | "M_N" => Some(fsci_constants::NEUTRON_MASS),
        "ATOMIC_MASS" | "U" => Some(fsci_constants::ATOMIC_MASS),
        "FINE_STRUCTURE" | "ALPHA" => Some(fsci_constants::FINE_STRUCTURE),
        "BOHR_RADIUS" => Some(fsci_constants::BOHR_RADIUS),
        "ELECTRON_VOLT" | "EV" => Some(fsci_constants::ELECTRON_VOLT),
        "CALORIE" => Some(fsci_constants::CALORIE),
        "ATMOSPHERE" | "ATM" => Some(fsci_constants::ATMOSPHERE),
        "BAR" => Some(fsci_constants::BAR),
        "POUND" => Some(fsci_constants::POUND),
        "INCH" => Some(fsci_constants::INCH),
        "FOOT" => Some(fsci_constants::FOOT),
        "DEGREE" => Some(fsci_constants::DEGREE),
        // br-wada: 22 additional CODATA physical constants (Faraday,
        // particle g-factors, Thomson cross section, characteristic
        // vacuum impedance, particle masses, molar / radiation /
        // ratio constants, Bohr magneton in eV/T).
        "FARADAY" => Some(fsci_constants::FARADAY),
        "ELECTRON_G_FACTOR" => Some(fsci_constants::ELECTRON_G_FACTOR),
        "PROTON_G_FACTOR" => Some(fsci_constants::PROTON_G_FACTOR),
        "NEUTRON_G_FACTOR" => Some(fsci_constants::NEUTRON_G_FACTOR),
        "MUON_G_FACTOR" => Some(fsci_constants::MUON_G_FACTOR),
        "THOMSON_CROSS_SECTION" => Some(fsci_constants::THOMSON_CROSS_SECTION),
        "CHARACTERISTIC_IMPEDANCE_OF_VACUUM" => {
            Some(fsci_constants::CHARACTERISTIC_IMPEDANCE_OF_VACUUM)
        }
        "DEUTERON_MASS" => Some(fsci_constants::DEUTERON_MASS),
        "ALPHA_PARTICLE_MASS" => Some(fsci_constants::ALPHA_PARTICLE_MASS),
        "MUON_MASS" => Some(fsci_constants::MUON_MASS),
        "TAU_MASS" => Some(fsci_constants::TAU_MASS),
        "HELION_MASS" => Some(fsci_constants::HELION_MASS),
        "TRITON_MASS" => Some(fsci_constants::TRITON_MASS),
        "MOLAR_VOLUME_IDEAL_GAS" => Some(fsci_constants::MOLAR_VOLUME_IDEAL_GAS),
        "MOLAR_PLANCK" => Some(fsci_constants::MOLAR_PLANCK),
        "RYDBERG_HZ" => Some(fsci_constants::RYDBERG_HZ),
        "INVERSE_FINE_STRUCTURE" => Some(fsci_constants::INVERSE_FINE_STRUCTURE),
        "FIRST_RADIATION_CONSTANT" => Some(fsci_constants::FIRST_RADIATION_CONSTANT),
        "SECOND_RADIATION_CONSTANT" => Some(fsci_constants::SECOND_RADIATION_CONSTANT),
        "ELECTRON_PROTON_MASS_RATIO" => Some(fsci_constants::ELECTRON_PROTON_MASS_RATIO),
        "PROTON_ELECTRON_MASS_RATIO" => Some(fsci_constants::PROTON_ELECTRON_MASS_RATIO),
        "BOHR_MAGNETON_EV_T" => Some(fsci_constants::BOHR_MAGNETON_EV_T),
        "COMPTON_WAVELENGTH" | "ELECTRON_COMPTON_WAVELENGTH" => {
            Some(fsci_constants::COMPTON_WAVELENGTH)
        }
        "ELECTRON_MASS_MEV" => Some(fsci_constants::ELECTRON_MASS_MEV),
        "PROTON_MASS_MEV" => Some(fsci_constants::PROTON_MASS_MEV),
        "NEUTRON_MASS_MEV" => Some(fsci_constants::NEUTRON_MASS_MEV),
        "MUON_MASS_MEV" => Some(fsci_constants::MUON_MASS_MEV),
        "TAU_MASS_MEV" => Some(fsci_constants::TAU_MASS_MEV),
        "DEUTERON_MASS_MEV" => Some(fsci_constants::DEUTERON_MASS_MEV),
        "ALPHA_PARTICLE_MASS_MEV" => Some(fsci_constants::ALPHA_PARTICLE_MASS_MEV),
        "HELION_MASS_MEV" => Some(fsci_constants::HELION_MASS_MEV),
        "TRITON_MASS_MEV" => Some(fsci_constants::TRITON_MASS_MEV),
        "PROTON_COMPTON_WAVELENGTH" => Some(fsci_constants::PROTON_COMPTON_WAVELENGTH),
        "NEUTRON_COMPTON_WAVELENGTH" => Some(fsci_constants::NEUTRON_COMPTON_WAVELENGTH),
        "MUON_COMPTON_WAVELENGTH" => Some(fsci_constants::MUON_COMPTON_WAVELENGTH),
        "TAU_COMPTON_WAVELENGTH" => Some(fsci_constants::TAU_COMPTON_WAVELENGTH),
        "NEUTRON_ELECTRON_MASS_RATIO" => Some(fsci_constants::NEUTRON_ELECTRON_MASS_RATIO),
        "ELECTRON_NEUTRON_MASS_RATIO" => Some(fsci_constants::ELECTRON_NEUTRON_MASS_RATIO),
        "MUON_ELECTRON_MASS_RATIO" => Some(fsci_constants::MUON_ELECTRON_MASS_RATIO),
        "ELECTRON_MUON_MASS_RATIO" => Some(fsci_constants::ELECTRON_MUON_MASS_RATIO),
        "PROTON_NEUTRON_MASS_RATIO" => Some(fsci_constants::PROTON_NEUTRON_MASS_RATIO),
        "NEUTRON_PROTON_MASS_RATIO" => Some(fsci_constants::NEUTRON_PROTON_MASS_RATIO),
        "DEUTERON_ELECTRON_MASS_RATIO" => Some(fsci_constants::DEUTERON_ELECTRON_MASS_RATIO),
        "ALPHA_PARTICLE_ELECTRON_MASS_RATIO" => {
            Some(fsci_constants::ALPHA_PARTICLE_ELECTRON_MASS_RATIO)
        }
        "HELION_ELECTRON_MASS_RATIO" => Some(fsci_constants::HELION_ELECTRON_MASS_RATIO),
        "TRITON_ELECTRON_MASS_RATIO" => Some(fsci_constants::TRITON_ELECTRON_MASS_RATIO),
        "TAU_ELECTRON_MASS_RATIO" => Some(fsci_constants::TAU_ELECTRON_MASS_RATIO),
        "ELECTRON_TAU_MASS_RATIO" => Some(fsci_constants::ELECTRON_TAU_MASS_RATIO),
        _ => None,
    }
}

fn execute_constants_case(case: &ConstantsCase) -> ConstantsObserved {
    fn arg_str(case: &ConstantsCase, idx: usize) -> Result<String, String> {
        case.args
            .get(idx)
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| format!("missing string arg[{idx}] for {}", case.function))
    }
    fn arg_f64(case: &ConstantsCase, idx: usize) -> Result<f64, String> {
        case.args
            .get(idx)
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| format!("missing f64 arg[{idx}] for {}", case.function))
    }

    match case.function.as_str() {
        "constant_value" => match arg_str(case, 0) {
            Ok(name) => match constant_value_by_name(&name) {
                Some(value) => ConstantsObserved::Scalar(value),
                None => ConstantsObserved::Error(format!("unknown constant: {name}")),
            },
            Err(e) => ConstantsObserved::Error(e),
        },
        "convert_temperature" => {
            let value = match arg_f64(case, 0) {
                Ok(v) => v,
                Err(e) => return ConstantsObserved::Error(e),
            };
            let from = match arg_str(case, 1) {
                Ok(v) => v,
                Err(e) => return ConstantsObserved::Error(e),
            };
            let to = match arg_str(case, 2) {
                Ok(v) => v,
                Err(e) => return ConstantsObserved::Error(e),
            };
            match fsci_constants::convert_temperature(value, &from, &to) {
                Ok(v) => ConstantsObserved::Scalar(v),
                Err(e) => ConstantsObserved::Error(e),
            }
        }
        "ev_to_joules" => match arg_f64(case, 0) {
            Ok(x) => ConstantsObserved::Scalar(fsci_constants::ev_to_joules(x)),
            Err(e) => ConstantsObserved::Error(e),
        },
        "joules_to_ev" => match arg_f64(case, 0) {
            Ok(x) => ConstantsObserved::Scalar(fsci_constants::joules_to_ev(x)),
            Err(e) => ConstantsObserved::Error(e),
        },
        "wavelength_to_freq" => match arg_f64(case, 0) {
            Ok(x) => ConstantsObserved::Scalar(fsci_constants::wavelength_to_freq(x)),
            Err(e) => ConstantsObserved::Error(e),
        },
        "freq_to_wavelength" => match arg_f64(case, 0) {
            Ok(x) => ConstantsObserved::Scalar(fsci_constants::freq_to_wavelength(x)),
            Err(e) => ConstantsObserved::Error(e),
        },
        "deg2rad" => match arg_f64(case, 0) {
            Ok(x) => ConstantsObserved::Scalar(fsci_constants::deg2rad(x)),
            Err(e) => ConstantsObserved::Error(e),
        },
        "rad2deg" => match arg_f64(case, 0) {
            Ok(x) => ConstantsObserved::Scalar(fsci_constants::rad2deg(x)),
            Err(e) => ConstantsObserved::Error(e),
        },
        "lb_to_kg" => match arg_f64(case, 0) {
            Ok(x) => ConstantsObserved::Scalar(fsci_constants::lb_to_kg(x)),
            Err(e) => ConstantsObserved::Error(e),
        },
        "kg_to_lb" => match arg_f64(case, 0) {
            Ok(x) => ConstantsObserved::Scalar(fsci_constants::kg_to_lb(x)),
            Err(e) => ConstantsObserved::Error(e),
        },
        other => ConstantsObserved::Error(format!("unknown function: {other}")),
    }
}

fn compare_constants_case_differential(
    case: &ConstantsCase,
    observed: &ConstantsObserved,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    let atol = case.expected.atol.unwrap_or(0.0);
    let rtol = case.expected.rtol.unwrap_or(0.0);
    let tolerance = ToleranceUsed {
        atol,
        rtol,
        comparison_mode: "allclose".to_owned(),
    };

    match (case.expected.kind.as_str(), observed) {
        ("scalar", ConstantsObserved::Scalar(actual)) => {
            let Some(expected) = case.expected.value else {
                return (
                    false,
                    "expected.value missing for scalar kind".to_owned(),
                    None,
                    Some(tolerance),
                );
            };
            let diff = (actual - expected).abs();
            let threshold = atol + rtol * expected.abs();
            let pass = diff <= threshold;
            let msg = if pass {
                format!("match within tolerance (diff={diff:e}, thr={threshold:e})")
            } else {
                format!(
                    "scalar mismatch: expected={expected:?}, actual={actual:?}, diff={diff:e}, thr={threshold:e}"
                )
            };
            (pass, msg, Some(diff), Some(tolerance))
        }
        ("error", ConstantsObserved::Error(actual)) => {
            let expected = case.expected.error.as_deref().unwrap_or("");
            let pass = matches_error_contract(actual, expected);
            let msg = if pass {
                "error matched".to_owned()
            } else {
                format!("error mismatch: expected `{expected}`, got `{actual}`")
            };
            (pass, msg, None, Some(tolerance))
        }
        (expected_kind, actual_obs) => (
            false,
            format!("shape mismatch: expected kind `{expected_kind}`, got {actual_obs:?}"),
            None,
            Some(tolerance),
        ),
    }
}

fn constants_oracle_case_to_expected(
    case: &ConstantsCase,
    oracle_case: &OracleCaseOutput,
) -> Result<ConstantsExpected, String> {
    let mut expected = ConstantsExpected {
        kind: oracle_case.result_kind.clone(),
        value: None,
        error: None,
        atol: case.expected.atol,
        rtol: case.expected.rtol,
        contract_ref: case.expected.contract_ref.clone(),
    };

    match oracle_case.result_kind.as_str() {
        "scalar" => {
            expected.value = Some(oracle_result_field(
                &oracle_case.result,
                "value",
                case.case_id(),
            )?);
        }
        "error" => {
            expected.error = Some(oracle_result_field(
                &oracle_case.result,
                "error",
                case.case_id(),
            )?);
        }
        other => {
            return Err(format!(
                "oracle result for {} has unsupported result_kind `{other}`",
                case.case_id()
            ));
        }
    }

    Ok(expected)
}

fn compare_constants_case_against_oracle(
    case: &ConstantsCase,
    oracle_case: &OracleCaseOutput,
    observed: &ConstantsObserved,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    if oracle_case.status != "ok" {
        return match observed {
            ConstantsObserved::Error(actual) => (
                false,
                format!(
                    "oracle errored (`{}`) and rust errored (`{actual}`) too; case is unjudgeable",
                    oracle_case
                        .error
                        .as_deref()
                        .unwrap_or("unknown oracle error"),
                ),
                None,
                None,
            ),
            ConstantsObserved::Scalar(_) => (
                false,
                format!(
                    "oracle errored (`{}`) but rust succeeded",
                    oracle_case
                        .error
                        .as_deref()
                        .unwrap_or("unknown oracle error"),
                ),
                None,
                None,
            ),
        };
    }

    match constants_oracle_case_to_expected(case, oracle_case) {
        Ok(expected) => {
            let mut oracle_case_fixture = case.clone();
            oracle_case_fixture.expected = expected;
            compare_constants_case_differential(&oracle_case_fixture, observed)
        }
        Err(message) => (false, message, None, None),
    }
}

fn run_differential_constants(
    fixture_path: &Path,
    raw: &str,
    oracle_config: &DifferentialOracleConfig,
) -> Result<ConformanceReport, HarnessError> {
    let fixture: ConstantsPacketFixture =
        serde_json::from_str(raw).map_err(|source| HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source,
        })?;

    let resolved_oracle_config = resolve_differential_oracle_config(oracle_config, &fixture.family);
    let probed_oracle_status = probe_oracle_availability(&resolved_oracle_config);
    let mut capture_failure_status = None;
    let oracle_capture = if resolved_oracle_config.required
        || matches!(probed_oracle_status, OracleStatus::Available)
    {
        match capture_python_oracle_inner(
            fixture_path,
            raw,
            &fixture.packet_id,
            &HarnessConfig::default_paths().oracle_root,
            &resolved_oracle_config,
        ) {
            Ok(capture) => Some(capture),
            Err(error) => {
                if resolved_oracle_config.required {
                    return Err(error);
                }
                capture_failure_status = Some(oracle_status_from_capture_error(&error));
                None
            }
        }
    } else {
        None
    };
    let oracle_cases = oracle_capture.as_ref().map(|capture| {
        capture
            .case_outputs
            .iter()
            .map(|case| (case.case_id.as_str(), case))
            .collect::<std::collections::HashMap<_, _>>()
    });
    let oracle_status = match (&oracle_capture, resolved_oracle_config.required) {
        (Some(_), _) => OracleStatus::Available,
        (None, true) => probed_oracle_status.clone(),
        (None, false) => capture_failure_status.unwrap_or(probed_oracle_status),
    };
    let mut per_case_results = Vec::with_capacity(fixture.cases.len());
    let audit_ledger = neutral_audit_ledger();

    for case in &fixture.cases {
        per_case_results.push(run_case_with_panic_capture(
            case.case_id(),
            &oracle_status,
            Some(audit_ledger.as_ref()),
            || {
                let observed = execute_constants_case(case);
                match oracle_cases.as_ref() {
                    Some(cases) => match cases.get(case.case_id()) {
                        Some(oracle_case) => {
                            compare_constants_case_against_oracle(case, oracle_case, &observed)
                        }
                        None => (
                            false,
                            format!("oracle capture missing constants case `{}`", case.case_id()),
                            None,
                            None,
                        ),
                    },
                    None => compare_constants_case_differential(case, &observed),
                }
            },
        ));
    }

    let pass_count = per_case_results
        .iter()
        .filter(|result| result.passed)
        .count();
    let fail_count = per_case_results.len().saturating_sub(pass_count);

    {
        let ledger = recover_sync_audit_ledger(audit_ledger.as_ref());
        let _ =
            emit_differential_audit_ledger_for_fixture(fixture_path, &fixture.packet_id, &ledger)?;
    }

    Ok(ConformanceReport {
        fixture_path: fixture_path.display().to_string(),
        packet_id: fixture.packet_id,
        family: fixture.family,
        pass_count,
        fail_count,
        oracle_status,
        per_case_results,
        generated_unix_ms: now_unix_ms(),
    })
}

fn execute_array_api_case(case: &ArrayApiCase) -> Result<ArrayApiObservedOutcome, String> {
    execute_array_api_case_inner(case, None)
}

fn execute_array_api_case_with_differential_audit(
    case: &ArrayApiCase,
    audit_ledger: &fsci_arrayapi::SyncSharedAuditLedger,
) -> Result<ArrayApiObservedOutcome, String> {
    execute_array_api_case_inner(case, Some(audit_ledger))
}

fn execute_array_api_case_inner(
    case: &ArrayApiCase,
    audit_ledger: Option<&fsci_arrayapi::SyncSharedAuditLedger>,
) -> Result<ArrayApiObservedOutcome, String> {
    let mode = array_api_exec_mode(match case {
        ArrayApiCase::Zeros { mode, .. }
        | ArrayApiCase::Ones { mode, .. }
        | ArrayApiCase::Full { mode, .. }
        | ArrayApiCase::Arange { mode, .. }
        | ArrayApiCase::Linspace { mode, .. }
        | ArrayApiCase::BroadcastShapes { mode, .. }
        | ArrayApiCase::ResultType { mode, .. }
        | ArrayApiCase::FromSlice { mode, .. }
        | ArrayApiCase::Getitem { mode, .. }
        | ArrayApiCase::Reshape { mode, .. }
        | ArrayApiCase::Transpose { mode, .. }
        | ArrayApiCase::RelationBroadcastCommutative { mode, .. }
        | ArrayApiCase::RelationResultTypeSymmetry { mode, .. }
        | ArrayApiCase::RelationIndexRoundtrip { mode, .. } => *mode,
    });
    let backend = FsciArrayApiCoreArrayBackend::new(mode);

    match case {
        ArrayApiCase::Zeros { shape, dtype, .. } => {
            let request = ArrayApiCreationRequest {
                shape: FsciArrayApiShape::new(shape.clone()),
                dtype: fixture_dtype_to_runtime(*dtype),
                order: FsciArrayApiMemoryOrder::C,
            };
            let array = arrayapi_zeros(&backend, &request).map_err(array_api_error_kind)?;
            Ok(observed_array(&array))
        }
        ArrayApiCase::Ones { shape, dtype, .. } => {
            let request = ArrayApiCreationRequest {
                shape: FsciArrayApiShape::new(shape.clone()),
                dtype: fixture_dtype_to_runtime(*dtype),
                order: FsciArrayApiMemoryOrder::C,
            };
            let array = arrayapi_ones(&backend, &request).map_err(array_api_error_kind)?;
            Ok(observed_array(&array))
        }
        ArrayApiCase::Full {
            shape,
            fill_value,
            dtype,
            ..
        } => {
            let request = ArrayApiFullRequest {
                fill_value: fixture_scalar_to_runtime(*fill_value),
                dtype: fixture_dtype_to_runtime(*dtype),
                order: FsciArrayApiMemoryOrder::C,
            };
            let array = arrayapi_full(&backend, &FsciArrayApiShape::new(shape.clone()), &request)
                .map_err(array_api_error_kind)?;
            Ok(observed_array(&array))
        }
        ArrayApiCase::Arange {
            start,
            stop,
            step,
            dtype,
            ..
        } => {
            let request = ArrayApiArangeRequest {
                start: fixture_scalar_to_runtime(*start),
                stop: fixture_scalar_to_runtime(*stop),
                step: fixture_scalar_to_runtime(*step),
                dtype: dtype.map(fixture_dtype_to_runtime),
            };
            let array = match audit_ledger {
                Some(ledger) => arrayapi_arange_with_audit(&backend, &request, ledger),
                None => arrayapi_arange(&backend, &request),
            }
            .map_err(array_api_error_kind)?;
            Ok(observed_array(&array))
        }
        ArrayApiCase::Linspace {
            start,
            stop,
            num,
            endpoint,
            dtype,
            ..
        } => {
            let request = ArrayApiLinspaceRequest {
                start: fixture_scalar_to_runtime(*start),
                stop: fixture_scalar_to_runtime(*stop),
                num: *num,
                endpoint: *endpoint,
                dtype: dtype.map(fixture_dtype_to_runtime),
            };
            let array = arrayapi_linspace(&backend, &request).map_err(array_api_error_kind)?;
            Ok(observed_array(&array))
        }
        ArrayApiCase::BroadcastShapes { shapes, .. } => {
            let runtime_shapes = shapes
                .iter()
                .map(|shape| FsciArrayApiShape::new(shape.clone()))
                .collect::<Vec<_>>();
            let out = match audit_ledger {
                Some(ledger) => arrayapi_broadcast_shapes_with_audit(&runtime_shapes, ledger),
                None => arrayapi_broadcast_shapes(&runtime_shapes),
            }
            .map_err(array_api_error_kind)?;
            Ok(ArrayApiObservedOutcome::Shape { dims: out.dims })
        }
        ArrayApiCase::ResultType {
            dtypes,
            force_floating,
            ..
        } => {
            let runtime_dtypes = dtypes
                .iter()
                .map(|dtype| fixture_dtype_to_runtime(*dtype))
                .collect::<Vec<_>>();
            let dtype = arrayapi_result_type(&backend, &runtime_dtypes, *force_floating)
                .map_err(array_api_error_kind)?;
            Ok(ArrayApiObservedOutcome::Dtype {
                dtype: runtime_dtype_to_fixture(dtype),
            })
        }
        ArrayApiCase::FromSlice {
            values,
            shape,
            dtype,
            ..
        } => {
            let request = ArrayApiCreationRequest {
                shape: FsciArrayApiShape::new(shape.clone()),
                dtype: fixture_dtype_to_runtime(*dtype),
                order: FsciArrayApiMemoryOrder::C,
            };
            let runtime_values = values
                .iter()
                .map(|value| fixture_scalar_to_runtime(*value))
                .collect::<Vec<_>>();
            let array = match audit_ledger {
                Some(ledger) => {
                    arrayapi_from_slice_with_audit(&backend, &runtime_values, &request, ledger)
                }
                None => arrayapi_from_slice(&backend, &runtime_values, &request),
            }
            .map_err(array_api_error_kind)?;
            Ok(observed_array(&array))
        }
        ArrayApiCase::Getitem {
            source_values,
            source_shape,
            source_dtype,
            indexing_mode,
            index,
            ..
        } => {
            let source = array_api_source_array(
                &backend,
                source_values,
                source_shape,
                *source_dtype,
                FsciArrayApiMemoryOrder::C,
            )?;
            let request = ArrayApiIndexRequest {
                mode: fixture_indexing_mode_to_runtime(*indexing_mode),
                index: fixture_index_to_runtime(index),
            };
            let array = match audit_ledger {
                Some(ledger) => arrayapi_getitem_with_audit(&backend, &source, &request, ledger),
                None => arrayapi_getitem(&backend, &source, &request),
            }
            .map_err(array_api_error_kind)?;
            Ok(observed_array(&array))
        }
        ArrayApiCase::Reshape {
            source_values,
            source_shape,
            source_dtype,
            new_shape,
            ..
        } => {
            let source = array_api_source_array(
                &backend,
                source_values,
                source_shape,
                *source_dtype,
                FsciArrayApiMemoryOrder::C,
            )?;
            let runtime_shape = FsciArrayApiShape::new(new_shape.clone());
            let array = match audit_ledger {
                Some(ledger) => {
                    arrayapi_reshape_with_audit(&backend, &source, &runtime_shape, ledger)
                }
                None => arrayapi_reshape(&backend, &source, &runtime_shape),
            }
            .map_err(array_api_error_kind)?;
            Ok(observed_array(&array))
        }
        ArrayApiCase::Transpose {
            source_values,
            source_shape,
            source_dtype,
            ..
        } => {
            let source = array_api_source_array(
                &backend,
                source_values,
                source_shape,
                *source_dtype,
                FsciArrayApiMemoryOrder::C,
            )?;
            let array = arrayapi_transpose(&backend, &source).map_err(array_api_error_kind)?;
            Ok(observed_array(&array))
        }
        ArrayApiCase::RelationBroadcastCommutative {
            left_shape,
            right_shape,
            ..
        } => {
            let left = FsciArrayApiShape::new(left_shape.clone());
            let right = FsciArrayApiShape::new(right_shape.clone());
            let lr = arrayapi_broadcast_shapes(&[left.clone(), right.clone()]);
            let rl = arrayapi_broadcast_shapes(&[right, left]);
            let pass = match (lr, rl) {
                (Ok(lhs), Ok(rhs)) => lhs == rhs,
                (Err(_), Err(_)) => true,
                _ => false,
            };
            Ok(ArrayApiObservedOutcome::Bool { value: pass })
        }
        ArrayApiCase::RelationResultTypeSymmetry {
            left_dtype,
            right_dtype,
            force_floating,
            ..
        } => {
            let left = fixture_dtype_to_runtime(*left_dtype);
            let right = fixture_dtype_to_runtime(*right_dtype);
            let forward = arrayapi_result_type(&backend, &[left, right], *force_floating);
            let reverse = arrayapi_result_type(&backend, &[right, left], *force_floating);
            let pass = match (forward, reverse) {
                (Ok(a), Ok(b)) => a == b,
                _ => false,
            };
            Ok(ArrayApiObservedOutcome::Bool { value: pass })
        }
        ArrayApiCase::RelationIndexRoundtrip {
            values,
            dtype,
            index,
            ..
        } => {
            let len = values.len();
            let source = array_api_source_array(
                &backend,
                values,
                &[len],
                *dtype,
                FsciArrayApiMemoryOrder::C,
            )?;
            let request = ArrayApiIndexRequest {
                mode: FsciArrayApiIndexingMode::Advanced,
                index: FsciArrayApiIndexExpr::Advanced {
                    indices: vec![vec![*index]],
                },
            };
            let selected =
                arrayapi_getitem(&backend, &source, &request).map_err(array_api_error_kind)?;
            let normalized = if *index < 0 {
                (len as isize + *index) as usize
            } else {
                *index as usize
            };
            let expected_value = fixture_scalar_to_runtime(values[normalized]);
            let pass = selected.size() == 1
                && selected
                    .values()
                    .first()
                    .is_some_and(|value| *value == expected_value);
            Ok(ArrayApiObservedOutcome::Bool { value: pass })
        }
    }
}

fn compare_array_api_case_differential(
    case: &ArrayApiCase,
    observed: &Result<ArrayApiObservedOutcome, String>,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    compare_array_api_expected_outcome(case.expected(), observed)
}

fn compare_array_api_expected_outcome(
    expected: &ArrayApiExpectedOutcome,
    observed: &Result<ArrayApiObservedOutcome, String>,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    match (expected, observed) {
        (
            ArrayApiExpectedOutcome::Array {
                shape,
                dtype,
                values,
                atol,
                rtol,
                contract_ref,
            },
            Ok(ArrayApiObservedOutcome::Array {
                shape: got_shape,
                dtype: got_dtype,
                values: got_values,
            }),
        ) => {
            let tolerance =
                resolve_array_api_contract_tolerance(contract_ref.as_deref(), *atol, *rtol);
            let shape_pass = got_shape == shape;
            let dtype_pass = got_dtype == dtype;
            let values_pass = got_values.len() == values.len()
                && got_values
                    .iter()
                    .zip(values.iter())
                    .all(|(actual, expected)| {
                        fixture_scalar_allclose(actual, expected, tolerance.atol, tolerance.rtol)
                    });
            let pass = shape_pass && dtype_pass && values_pass;
            let max_diff = fixture_max_diff_seq(got_values, values);
            let msg = if pass {
                format!(
                    "array matched shape={shape:?} dtype={dtype:?} (max_diff={:.2e})",
                    max_diff
                )
            } else {
                format!(
                    "array mismatch: expected shape={shape:?} dtype={dtype:?}, got shape={got_shape:?} dtype={got_dtype:?}"
                )
            };
            (pass, msg, Some(max_diff), Some(tolerance))
        }
        (
            ArrayApiExpectedOutcome::Shape { dims, contract_ref },
            Ok(ArrayApiObservedOutcome::Shape { dims: got }),
        ) => {
            let pass = got == dims;
            let msg = if pass {
                format!("shape matched {dims:?}")
            } else {
                format!("shape mismatch: expected {dims:?}, got {got:?}")
            };
            let tolerance = contract_ref
                .as_deref()
                .map(|value| resolve_array_api_contract_tolerance(Some(value), None, None));
            (pass, msg, None, tolerance)
        }
        (
            ArrayApiExpectedOutcome::Dtype {
                dtype,
                contract_ref,
            },
            Ok(ArrayApiObservedOutcome::Dtype { dtype: got }),
        ) => {
            let pass = got == dtype;
            let msg = if pass {
                format!("dtype matched {dtype:?}")
            } else {
                format!("dtype mismatch: expected {dtype:?}, got {got:?}")
            };
            let tolerance = contract_ref
                .as_deref()
                .map(|value| resolve_array_api_contract_tolerance(Some(value), None, None));
            (pass, msg, None, tolerance)
        }
        (
            ArrayApiExpectedOutcome::Bool {
                value,
                contract_ref,
            },
            Ok(ArrayApiObservedOutcome::Bool { value: got }),
        ) => {
            let pass = got == value;
            let msg = if pass {
                format!("relation matched ({value})")
            } else {
                format!("relation mismatch: expected={value}, got={got}")
            };
            let tolerance = contract_ref
                .as_deref()
                .map(|value| resolve_array_api_contract_tolerance(Some(value), None, None));
            (pass, msg, None, tolerance)
        }
        (ArrayApiExpectedOutcome::ErrorKind { error }, Err(observed_kind)) => {
            let pass = observed_kind == error;
            let msg = if pass {
                format!("error kind matched ({observed_kind})")
            } else {
                format!("error kind mismatch: expected={error}, got={observed_kind}")
            };
            (pass, msg, None, None)
        }
        (expected, observed) => (
            false,
            format!("shape mismatch: expected {expected:?}, got {observed:?}"),
            None,
            None,
        ),
    }
}

fn array_api_expected_metadata(
    expected: &ArrayApiExpectedOutcome,
) -> (Option<f64>, Option<f64>, Option<String>) {
    match expected {
        ArrayApiExpectedOutcome::Array {
            atol,
            rtol,
            contract_ref,
            ..
        } => (*atol, *rtol, contract_ref.clone()),
        ArrayApiExpectedOutcome::Shape { contract_ref, .. }
        | ArrayApiExpectedOutcome::Dtype { contract_ref, .. }
        | ArrayApiExpectedOutcome::Bool { contract_ref, .. } => (None, None, contract_ref.clone()),
        ArrayApiExpectedOutcome::ErrorKind { .. } => (None, None, None),
    }
}

fn array_api_oracle_case_to_expected(
    case: &ArrayApiCase,
    oracle_case: &OracleCaseOutput,
) -> Result<ArrayApiExpectedOutcome, String> {
    let (atol, rtol, contract_ref) = array_api_expected_metadata(case.expected());

    match oracle_case.result_kind.as_str() {
        "array" => Ok(ArrayApiExpectedOutcome::Array {
            shape: oracle_result_field(&oracle_case.result, "shape", case.case_id())?,
            dtype: oracle_result_field(&oracle_case.result, "dtype", case.case_id())?,
            values: oracle_result_field(&oracle_case.result, "values", case.case_id())?,
            atol,
            rtol,
            contract_ref,
        }),
        "shape" => Ok(ArrayApiExpectedOutcome::Shape {
            dims: oracle_result_field(&oracle_case.result, "dims", case.case_id())?,
            contract_ref,
        }),
        "dtype" => Ok(ArrayApiExpectedOutcome::Dtype {
            dtype: oracle_result_field(&oracle_case.result, "dtype", case.case_id())?,
            contract_ref,
        }),
        "bool" => Ok(ArrayApiExpectedOutcome::Bool {
            value: oracle_result_field(&oracle_case.result, "value", case.case_id())?,
            contract_ref,
        }),
        "error_kind" => Ok(ArrayApiExpectedOutcome::ErrorKind {
            error: oracle_result_field(&oracle_case.result, "error", case.case_id())?,
        }),
        other => Err(format!(
            "oracle result for {} has unsupported result_kind `{other}`",
            case.case_id()
        )),
    }
}

fn compare_array_api_case_against_oracle(
    case: &ArrayApiCase,
    oracle_case: &OracleCaseOutput,
    observed: &Result<ArrayApiObservedOutcome, String>,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    if oracle_case.status != "ok" {
        return match observed {
            Err(actual) => (
                false,
                format!(
                    "oracle errored (`{}`) and rust errored (`{actual}`) too; case is unjudgeable",
                    oracle_case
                        .error
                        .as_deref()
                        .unwrap_or("unknown oracle error"),
                ),
                None,
                None,
            ),
            Ok(_) => (
                false,
                format!(
                    "oracle errored (`{}`) but rust succeeded",
                    oracle_case
                        .error
                        .as_deref()
                        .unwrap_or("unknown oracle error"),
                ),
                None,
                None,
            ),
        };
    }

    match array_api_oracle_case_to_expected(case, oracle_case) {
        Ok(expected) => compare_array_api_expected_outcome(&expected, observed),
        Err(message) => (false, message, None, None),
    }
}

fn array_api_source_array(
    backend: &FsciArrayApiCoreArrayBackend,
    values: &[ArrayApiFixtureScalar],
    shape: &[usize],
    dtype: ArrayApiFixtureDType,
    order: FsciArrayApiMemoryOrder,
) -> Result<FsciArrayApiCoreArray, String> {
    let request = ArrayApiCreationRequest {
        shape: FsciArrayApiShape::new(shape.to_vec()),
        dtype: fixture_dtype_to_runtime(dtype),
        order,
    };
    let runtime_values = values
        .iter()
        .map(|value| fixture_scalar_to_runtime(*value))
        .collect::<Vec<_>>();
    arrayapi_from_slice(backend, &runtime_values, &request).map_err(array_api_error_kind)
}

fn observed_array(array: &FsciArrayApiCoreArray) -> ArrayApiObservedOutcome {
    use fsci_arrayapi::MemoryOrder;

    let values = if array.order() == MemoryOrder::F && array.shape().rank() == 2 {
        let rows = array.shape().dims[0];
        let cols = array.shape().dims[1];
        let mut row_major = Vec::with_capacity(array.values().len());
        for r in 0..rows {
            for c in 0..cols {
                // In F-order, index is c * rows + r
                row_major.push(array.values()[c * rows + r]);
            }
        }
        row_major
    } else {
        array.values().to_vec()
    };

    ArrayApiObservedOutcome::Array {
        shape: array.shape().dims.clone(),
        dtype: runtime_dtype_to_fixture(array.dtype()),
        values: values.into_iter().map(runtime_scalar_to_fixture).collect(),
    }
}

fn array_api_exec_mode(mode: RuntimeMode) -> FsciArrayApiExecutionMode {
    match mode {
        RuntimeMode::Strict => FsciArrayApiExecutionMode::Strict,
        RuntimeMode::Hardened => FsciArrayApiExecutionMode::Hardened,
    }
}

fn fixture_dtype_to_runtime(dtype: ArrayApiFixtureDType) -> FsciArrayApiDType {
    match dtype {
        ArrayApiFixtureDType::Bool => FsciArrayApiDType::Bool,
        ArrayApiFixtureDType::Int64 => FsciArrayApiDType::Int64,
        ArrayApiFixtureDType::UInt64 => FsciArrayApiDType::UInt64,
        ArrayApiFixtureDType::Float32 => FsciArrayApiDType::Float32,
        ArrayApiFixtureDType::Float64 => FsciArrayApiDType::Float64,
        ArrayApiFixtureDType::Complex64 => FsciArrayApiDType::Complex64,
        ArrayApiFixtureDType::Complex128 => FsciArrayApiDType::Complex128,
    }
}

fn runtime_dtype_to_fixture(dtype: FsciArrayApiDType) -> ArrayApiFixtureDType {
    match dtype {
        FsciArrayApiDType::Bool => ArrayApiFixtureDType::Bool,
        FsciArrayApiDType::Int64 => ArrayApiFixtureDType::Int64,
        FsciArrayApiDType::UInt64 => ArrayApiFixtureDType::UInt64,
        FsciArrayApiDType::Float32 => ArrayApiFixtureDType::Float32,
        FsciArrayApiDType::Float64 => ArrayApiFixtureDType::Float64,
        FsciArrayApiDType::Complex64 => ArrayApiFixtureDType::Complex64,
        FsciArrayApiDType::Complex128 => ArrayApiFixtureDType::Complex128,
        FsciArrayApiDType::Int8
        | FsciArrayApiDType::Int16
        | FsciArrayApiDType::Int32
        | FsciArrayApiDType::UInt8
        | FsciArrayApiDType::UInt16
        | FsciArrayApiDType::UInt32 => ArrayApiFixtureDType::Int64,
    }
}

fn fixture_scalar_to_runtime(value: ArrayApiFixtureScalar) -> FsciArrayApiScalarValue {
    match value {
        ArrayApiFixtureScalar::Bool { value } => FsciArrayApiScalarValue::Bool(value),
        ArrayApiFixtureScalar::I64 { value } => FsciArrayApiScalarValue::I64(value),
        ArrayApiFixtureScalar::U64 { value } => FsciArrayApiScalarValue::U64(value),
        ArrayApiFixtureScalar::F64 { value } => FsciArrayApiScalarValue::F64(value),
        ArrayApiFixtureScalar::ComplexF64 { re, im } => {
            FsciArrayApiScalarValue::ComplexF64 { re, im }
        }
    }
}

fn runtime_scalar_to_fixture(value: FsciArrayApiScalarValue) -> ArrayApiFixtureScalar {
    match value {
        FsciArrayApiScalarValue::Bool(value) => ArrayApiFixtureScalar::Bool { value },
        FsciArrayApiScalarValue::I64(value) => ArrayApiFixtureScalar::I64 { value },
        FsciArrayApiScalarValue::U64(value) => ArrayApiFixtureScalar::U64 { value },
        FsciArrayApiScalarValue::F64(value) => ArrayApiFixtureScalar::F64 { value },
        FsciArrayApiScalarValue::ComplexF64 { re, im } => {
            ArrayApiFixtureScalar::ComplexF64 { re, im }
        }
    }
}

fn fixture_indexing_mode_to_runtime(mode: ArrayApiFixtureIndexingMode) -> FsciArrayApiIndexingMode {
    match mode {
        ArrayApiFixtureIndexingMode::Basic => FsciArrayApiIndexingMode::Basic,
        ArrayApiFixtureIndexingMode::Advanced => FsciArrayApiIndexingMode::Advanced,
        ArrayApiFixtureIndexingMode::BooleanMask => FsciArrayApiIndexingMode::BooleanMask,
    }
}

fn fixture_index_to_runtime(index: &ArrayApiFixtureIndexExpr) -> FsciArrayApiIndexExpr {
    match index {
        ArrayApiFixtureIndexExpr::Basic { slices } => FsciArrayApiIndexExpr::Basic {
            slices: slices
                .iter()
                .map(|slice| FsciArrayApiSliceSpec {
                    start: slice.start,
                    stop: slice.stop,
                    step: slice.step,
                })
                .collect(),
        },
        ArrayApiFixtureIndexExpr::Advanced { indices } => FsciArrayApiIndexExpr::Advanced {
            indices: indices.clone(),
        },
        ArrayApiFixtureIndexExpr::BooleanMask { mask_shape } => {
            FsciArrayApiIndexExpr::BooleanMask {
                mask_shape: FsciArrayApiShape::new(mask_shape.clone()),
            }
        }
    }
}

fn array_api_error_kind(error: fsci_arrayapi::ArrayApiError) -> String {
    format!("{:?}", error.kind)
}

fn fixture_scalar_diff(actual: &ArrayApiFixtureScalar, expected: &ArrayApiFixtureScalar) -> f64 {
    match (actual, expected) {
        (ArrayApiFixtureScalar::Bool { value: a }, ArrayApiFixtureScalar::Bool { value: e }) => {
            if a == e { 0.0 } else { 1.0 }
        }
        (ArrayApiFixtureScalar::I64 { value: a }, ArrayApiFixtureScalar::I64 { value: e }) => {
            (*a as f64 - *e as f64).abs()
        }
        (ArrayApiFixtureScalar::U64 { value: a }, ArrayApiFixtureScalar::U64 { value: e }) => {
            (*a as f64 - *e as f64).abs()
        }
        (ArrayApiFixtureScalar::F64 { value: a }, ArrayApiFixtureScalar::F64 { value: e }) => {
            (*a - *e).abs()
        }
        (
            ArrayApiFixtureScalar::ComplexF64 { re: ar, im: ai },
            ArrayApiFixtureScalar::ComplexF64 { re: er, im: ei },
        ) => (ar - er).abs().max((ai - ei).abs()),
        _ => f64::INFINITY,
    }
}

fn fixture_scalar_allclose(
    actual: &ArrayApiFixtureScalar,
    expected: &ArrayApiFixtureScalar,
    atol: f64,
    rtol: f64,
) -> bool {
    match (actual, expected) {
        (ArrayApiFixtureScalar::Bool { value: a }, ArrayApiFixtureScalar::Bool { value: e }) => {
            a == e
        }
        (ArrayApiFixtureScalar::I64 { value: a }, ArrayApiFixtureScalar::I64 { value: e }) => {
            a == e
        }
        (ArrayApiFixtureScalar::U64 { value: a }, ArrayApiFixtureScalar::U64 { value: e }) => {
            a == e
        }
        (ArrayApiFixtureScalar::F64 { value: a }, ArrayApiFixtureScalar::F64 { value: e }) => {
            allclose_scalar(*a, *e, atol, rtol)
        }
        (
            ArrayApiFixtureScalar::ComplexF64 { re: ar, im: ai },
            ArrayApiFixtureScalar::ComplexF64 { re: er, im: ei },
        ) => allclose_scalar(*ar, *er, atol, rtol) && allclose_scalar(*ai, *ei, atol, rtol),
        _ => false,
    }
}

fn fixture_max_diff_seq(
    actual: &[ArrayApiFixtureScalar],
    expected: &[ArrayApiFixtureScalar],
) -> f64 {
    if actual.len() != expected.len() {
        return f64::INFINITY;
    }
    actual
        .iter()
        .zip(expected.iter())
        .map(|(a, e)| fixture_scalar_diff(a, e))
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        })
}

fn execute_special_case_scalar(case: &SpecialCase) -> Result<f64, FsciSpecialError> {
    let mode = case.mode;
    let args = case.real_scalar_args().ok_or(FsciSpecialError {
        function: "special_fixture",
        kind: FsciSpecialErrorKind::DomainError,
        mode,
        detail: "special scalar path requires real scalar fixture args",
    })?;
    match case.function {
        SpecialCaseFunction::Gamma => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("gamma", mode));
            }
            special_scalar_from_tensor(
                special_gamma(&special_scalar(args[0]), mode)?,
                "gamma",
                mode,
            )
        }
        SpecialCaseFunction::Gammaln => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("gammaln", mode));
            }
            special_scalar_from_tensor(
                special_gammaln(&special_scalar(args[0]), mode)?,
                "gammaln",
                mode,
            )
        }
        SpecialCaseFunction::Multigammaln => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("multigammaln", mode));
            }
            special_scalar_from_tensor(
                special_multigammaln(&special_scalar(args[0]), args[1], mode)?,
                "multigammaln",
                mode,
            )
        }
        SpecialCaseFunction::Digamma => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("digamma", mode));
            }
            special_scalar_from_tensor(
                special_digamma(&special_scalar(args[0]), mode)?,
                "digamma",
                mode,
            )
        }
        SpecialCaseFunction::Polygamma => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("polygamma", mode));
            }
            let order = special_usize_from_fixture("polygamma", mode, args[0])?;
            special_scalar_from_tensor(
                special_polygamma(order, &special_scalar(args[1]), mode)?,
                "polygamma",
                mode,
            )
        }
        SpecialCaseFunction::Rgamma => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("rgamma", mode));
            }
            special_scalar_from_tensor(
                special_rgamma(&special_scalar(args[0]), mode)?,
                "rgamma",
                mode,
            )
        }
        SpecialCaseFunction::Gammainc => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("gammainc", mode));
            }
            special_scalar_from_tensor(
                special_gammainc(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "gammainc",
                mode,
            )
        }
        SpecialCaseFunction::Gammaincc => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("gammaincc", mode));
            }
            special_scalar_from_tensor(
                special_gammaincc(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "gammaincc",
                mode,
            )
        }
        SpecialCaseFunction::Factorial => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("factorial", mode));
            }
            Ok(special_factorial(special_u64_from_fixture(
                "factorial",
                mode,
                args[0],
            )?))
        }
        SpecialCaseFunction::Factorial2 => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("factorial2", mode));
            }
            Ok(special_factorial2(special_i64_from_fixture(
                "factorial2",
                mode,
                args[0],
            )?))
        }
        SpecialCaseFunction::Comb => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("comb", mode));
            }
            Ok(special_comb(
                special_u64_from_fixture("comb", mode, args[0])?,
                special_u64_from_fixture("comb", mode, args[1])?,
            ))
        }
        SpecialCaseFunction::Perm => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("perm", mode));
            }
            Ok(special_perm(
                special_u64_from_fixture("perm", mode, args[0])?,
                special_u64_from_fixture("perm", mode, args[1])?,
            ))
        }
        SpecialCaseFunction::Poch => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("poch", mode));
            }
            Ok(special_poch(args[0], args[1]))
        }
        SpecialCaseFunction::Erf => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("erf", mode));
            }
            special_scalar_from_tensor(special_erf(&special_scalar(args[0]), mode)?, "erf", mode)
        }
        SpecialCaseFunction::Erfc => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("erfc", mode));
            }
            special_scalar_from_tensor(special_erfc(&special_scalar(args[0]), mode)?, "erfc", mode)
        }
        SpecialCaseFunction::Erfinv => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("erfinv", mode));
            }
            special_scalar_from_tensor(
                special_erfinv(&special_scalar(args[0]), mode)?,
                "erfinv",
                mode,
            )
        }
        SpecialCaseFunction::Erfcinv => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("erfcinv", mode));
            }
            special_scalar_from_tensor(
                special_erfcinv(&special_scalar(args[0]), mode)?,
                "erfcinv",
                mode,
            )
        }
        SpecialCaseFunction::Erfcx => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("erfcx", mode));
            }
            special_scalar_from_tensor(
                special_erfcx(&special_scalar(args[0]), mode)?,
                "erfcx",
                mode,
            )
        }
        SpecialCaseFunction::Erfi => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("erfi", mode));
            }
            special_scalar_from_tensor(special_erfi(&special_scalar(args[0]), mode)?, "erfi", mode)
        }
        SpecialCaseFunction::OwensT => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("owens_t", mode));
            }
            special_scalar_from_tensor(
                special_owens_t(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "owens_t",
                mode,
            )
        }
        SpecialCaseFunction::Beta => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("beta", mode));
            }
            special_scalar_from_tensor(
                special_beta(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "beta",
                mode,
            )
        }
        SpecialCaseFunction::Betaln => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("betaln", mode));
            }
            special_scalar_from_tensor(
                special_betaln(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "betaln",
                mode,
            )
        }
        SpecialCaseFunction::Betainc => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("betainc", mode));
            }
            special_scalar_from_tensor(
                special_betainc(
                    &special_scalar(args[0]),
                    &special_scalar(args[1]),
                    &special_scalar(args[2]),
                    mode,
                )?,
                "betainc",
                mode,
            )
        }
        SpecialCaseFunction::Btdtr => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("btdtr", mode));
            }
            Ok(special_btdtr(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Btdtrc => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("btdtrc", mode));
            }
            Ok(special_btdtrc(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Btdtri => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("btdtri", mode));
            }
            Ok(special_btdtri(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Btdtria => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("btdtria", mode));
            }
            Ok(special_btdtria(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Btdtrib => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("btdtrib", mode));
            }
            Ok(special_btdtrib(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Fdtr => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("fdtr", mode));
            }
            Ok(special_fdtr(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Fdtrc => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("fdtrc", mode));
            }
            Ok(special_fdtrc(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Fdtri => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("fdtri", mode));
            }
            Ok(special_fdtri(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Fdtridfd => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("fdtridfd", mode));
            }
            Ok(special_fdtridfd(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Stdtr => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("stdtr", mode));
            }
            Ok(special_stdtr(args[0], args[1]))
        }
        SpecialCaseFunction::Stdtrc => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("stdtrc", mode));
            }
            Ok(special_stdtrc(args[0], args[1]))
        }
        SpecialCaseFunction::Stdtridf => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("stdtridf", mode));
            }
            Ok(special_stdtridf(args[0], args[1]))
        }
        SpecialCaseFunction::Stdtrit => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("stdtrit", mode));
            }
            Ok(special_stdtrit(args[0], args[1]))
        }
        SpecialCaseFunction::Bdtr => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("bdtr", mode));
            }
            Ok(special_bdtr(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Bdtrc => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("bdtrc", mode));
            }
            Ok(special_bdtrc(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Bdtri => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("bdtri", mode));
            }
            Ok(special_bdtri(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Nbdtr => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("nbdtr", mode));
            }
            Ok(special_nbdtr(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Nbdtrc => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("nbdtrc", mode));
            }
            Ok(special_nbdtrc(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Nbdtri => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("nbdtri", mode));
            }
            Ok(special_nbdtri(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Pdtr => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("pdtr", mode));
            }
            Ok(special_pdtr(args[0], args[1]))
        }
        SpecialCaseFunction::Pdtrc => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("pdtrc", mode));
            }
            Ok(special_pdtrc(args[0], args[1]))
        }
        SpecialCaseFunction::Pdtri => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("pdtri", mode));
            }
            Ok(special_pdtri(args[0], args[1]))
        }
        SpecialCaseFunction::Pdtrik => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("pdtrik", mode));
            }
            Ok(special_pdtrik(args[0], args[1]))
        }
        SpecialCaseFunction::Chdtr => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("chdtr", mode));
            }
            Ok(special_chdtr(args[0], args[1]))
        }
        SpecialCaseFunction::Chdtrc => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("chdtrc", mode));
            }
            Ok(special_chdtrc(args[0], args[1]))
        }
        SpecialCaseFunction::Chdtri => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("chdtri", mode));
            }
            Ok(special_chdtri(args[0], args[1]))
        }
        SpecialCaseFunction::Chdtriv => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("chdtriv", mode));
            }
            Ok(special_chdtriv(args[0], args[1]))
        }
        SpecialCaseFunction::Gdtr => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("gdtr", mode));
            }
            Ok(special_gdtr(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Gdtrc => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("gdtrc", mode));
            }
            Ok(special_gdtrc(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Gdtria => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("gdtria", mode));
            }
            Ok(special_gdtria(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Gdtrib => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("gdtrib", mode));
            }
            Ok(special_gdtrib(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Gdtrix => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("gdtrix", mode));
            }
            Ok(special_gdtrix(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Ellipk => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("ellipk", mode));
            }
            special_scalar_from_tensor(
                special_ellipk(&special_scalar(args[0]), mode)?,
                "ellipk",
                mode,
            )
        }
        SpecialCaseFunction::Ellipkm1 => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("ellipkm1", mode));
            }
            special_scalar_from_tensor(
                special_ellipkm1(&special_scalar(args[0]), mode)?,
                "ellipkm1",
                mode,
            )
        }
        SpecialCaseFunction::Ellipe => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("ellipe", mode));
            }
            special_scalar_from_tensor(
                special_ellipe(&special_scalar(args[0]), mode)?,
                "ellipe",
                mode,
            )
        }
        SpecialCaseFunction::Ellipkinc => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("ellipkinc", mode));
            }
            special_scalar_from_tensor(
                special_ellipkinc(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "ellipkinc",
                mode,
            )
        }
        SpecialCaseFunction::Ellipeinc => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("ellipeinc", mode));
            }
            special_scalar_from_tensor(
                special_ellipeinc(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "ellipeinc",
                mode,
            )
        }
        SpecialCaseFunction::EllipjSn => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("ellipj_sn", mode));
            }
            let (sn, _, _, _) = special_ellipj(args[0], args[1]);
            Ok(sn)
        }
        SpecialCaseFunction::EllipjCn => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("ellipj_cn", mode));
            }
            let (_, cn, _, _) = special_ellipj(args[0], args[1]);
            Ok(cn)
        }
        SpecialCaseFunction::EllipjDn => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("ellipj_dn", mode));
            }
            let (_, _, dn, _) = special_ellipj(args[0], args[1]);
            Ok(dn)
        }
        SpecialCaseFunction::EllipjPh => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("ellipj_ph", mode));
            }
            let (_, _, _, ph) = special_ellipj(args[0], args[1]);
            Ok(ph)
        }
        SpecialCaseFunction::Lambertw => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("lambertw", mode));
            }
            special_scalar_from_tensor(
                special_lambertw(&special_scalar(args[0]), mode)?,
                "lambertw",
                mode,
            )
        }
        SpecialCaseFunction::Exp1 => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("exp1", mode));
            }
            special_scalar_from_tensor(special_exp1(&special_scalar(args[0]), mode)?, "exp1", mode)
        }
        SpecialCaseFunction::Expi => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("expi", mode));
            }
            special_scalar_from_tensor(special_expi(&special_scalar(args[0]), mode)?, "expi", mode)
        }
        SpecialCaseFunction::Expn => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("expn", mode));
            }
            let order = special_u32_from_fixture("expn", mode, args[0])?;
            Ok(special_expn(order, args[1]))
        }
        SpecialCaseFunction::Hyp0f1 => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("hyp0f1", mode));
            }
            special_scalar_from_tensor(
                special_hyp0f1(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "hyp0f1",
                mode,
            )
        }
        SpecialCaseFunction::Hyp1f1 => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("hyp1f1", mode));
            }
            special_scalar_from_tensor(
                special_hyp1f1(
                    &special_scalar(args[0]),
                    &special_scalar(args[1]),
                    &special_scalar(args[2]),
                    mode,
                )?,
                "hyp1f1",
                mode,
            )
        }
        SpecialCaseFunction::Hyp2f1 => {
            if args.len() != 4 {
                return Err(special_invalid_fixture_error("hyp2f1", mode));
            }
            special_scalar_from_tensor(
                special_hyp2f1(
                    &special_scalar(args[0]),
                    &special_scalar(args[1]),
                    &special_scalar(args[2]),
                    &special_scalar(args[3]),
                    mode,
                )?,
                "hyp2f1",
                mode,
            )
        }
        SpecialCaseFunction::J0 => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("j0", mode));
            }
            special_scalar_from_tensor(special_j0(&special_scalar(args[0]), mode)?, "j0", mode)
        }
        SpecialCaseFunction::J1 => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("j1", mode));
            }
            special_scalar_from_tensor(special_j1(&special_scalar(args[0]), mode)?, "j1", mode)
        }
        SpecialCaseFunction::Jn => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("jn", mode));
            }
            special_scalar_from_tensor(
                special_jn(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "jn",
                mode,
            )
        }
        SpecialCaseFunction::Y0 => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("y0", mode));
            }
            special_scalar_from_tensor(special_y0(&special_scalar(args[0]), mode)?, "y0", mode)
        }
        SpecialCaseFunction::Y1 => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("y1", mode));
            }
            special_scalar_from_tensor(special_y1(&special_scalar(args[0]), mode)?, "y1", mode)
        }
        SpecialCaseFunction::Yn => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("yn", mode));
            }
            special_scalar_from_tensor(
                special_yn(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "yn",
                mode,
            )
        }
        SpecialCaseFunction::Iv => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("iv", mode));
            }
            special_scalar_from_tensor(
                special_iv(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "iv",
                mode,
            )
        }
        SpecialCaseFunction::Kv => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("kv", mode));
            }
            special_scalar_from_tensor(
                special_kv(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "kv",
                mode,
            )
        }
        SpecialCaseFunction::WrightBessel => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("wright_bessel", mode));
            }
            special_scalar_from_tensor(
                special_wright_bessel(
                    &special_scalar(args[0]),
                    &special_scalar(args[1]),
                    &special_scalar(args[2]),
                    mode,
                )?,
                "wright_bessel",
                mode,
            )
        }
        SpecialCaseFunction::Jvp => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("jvp", mode));
            }
            let derivative_order = special_derivative_order_from_fixture("jvp", mode, args[2])?;
            special_scalar_from_tensor(
                special_jvp(
                    &special_scalar(args[0]),
                    &special_scalar(args[1]),
                    derivative_order,
                    mode,
                )?,
                "jvp",
                mode,
            )
        }
        SpecialCaseFunction::Yvp => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("yvp", mode));
            }
            let derivative_order = special_derivative_order_from_fixture("yvp", mode, args[2])?;
            special_scalar_from_tensor(
                special_yvp(
                    &special_scalar(args[0]),
                    &special_scalar(args[1]),
                    derivative_order,
                    mode,
                )?,
                "yvp",
                mode,
            )
        }
        SpecialCaseFunction::Ivp => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("ivp", mode));
            }
            let derivative_order = special_derivative_order_from_fixture("ivp", mode, args[2])?;
            special_scalar_from_tensor(
                special_ivp(
                    &special_scalar(args[0]),
                    &special_scalar(args[1]),
                    derivative_order,
                    mode,
                )?,
                "ivp",
                mode,
            )
        }
        SpecialCaseFunction::Kvp => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("kvp", mode));
            }
            let derivative_order = special_derivative_order_from_fixture("kvp", mode, args[2])?;
            special_scalar_from_tensor(
                special_kvp(
                    &special_scalar(args[0]),
                    &special_scalar(args[1]),
                    derivative_order,
                    mode,
                )?,
                "kvp",
                mode,
            )
        }
        SpecialCaseFunction::SphericalJn => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("spherical_jn", mode));
            }
            special_scalar_from_tensor(
                special_spherical_jn(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "spherical_jn",
                mode,
            )
        }
        SpecialCaseFunction::SphericalYn => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("spherical_yn", mode));
            }
            special_scalar_from_tensor(
                special_spherical_yn(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "spherical_yn",
                mode,
            )
        }
        SpecialCaseFunction::SphericalIn => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("spherical_in", mode));
            }
            special_scalar_from_tensor(
                special_spherical_in(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "spherical_in",
                mode,
            )
        }
        SpecialCaseFunction::SphericalKn => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("spherical_kn", mode));
            }
            special_scalar_from_tensor(
                special_spherical_kn(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "spherical_kn",
                mode,
            )
        }
        SpecialCaseFunction::Sinc => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("sinc", mode));
            }
            special_scalar_from_tensor(special_sinc(&special_scalar(args[0]), mode)?, "sinc", mode)
        }
        SpecialCaseFunction::Xlogy => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("xlogy", mode));
            }
            special_scalar_from_tensor(
                special_xlogy(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "xlogy",
                mode,
            )
        }
        SpecialCaseFunction::Xlog1py => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("xlog1py", mode));
            }
            special_scalar_from_tensor(
                special_xlog1py(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "xlog1py",
                mode,
            )
        }
        SpecialCaseFunction::Xlogx => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("xlogx", mode));
            }
            special_scalar_from_tensor(
                special_xlogx(&special_scalar(args[0]), mode)?,
                "xlogx",
                mode,
            )
        }
        SpecialCaseFunction::Kolmogorov => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("kolmogorov", mode));
            }
            special_scalar_from_tensor(
                special_kolmogorov(&special_scalar(args[0]), mode)?,
                "kolmogorov",
                mode,
            )
        }
        SpecialCaseFunction::Kolmogi => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("kolmogi", mode));
            }
            special_scalar_from_tensor(
                special_kolmogi(&special_scalar(args[0]), mode)?,
                "kolmogi",
                mode,
            )
        }
        SpecialCaseFunction::Entr => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("entr", mode));
            }
            special_scalar_from_tensor(special_entr(&special_scalar(args[0]), mode)?, "entr", mode)
        }
        SpecialCaseFunction::RelEntr => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("rel_entr", mode));
            }
            special_scalar_from_tensor(
                special_rel_entr(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "rel_entr",
                mode,
            )
        }
        SpecialCaseFunction::KlDiv => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("kl_div", mode));
            }
            Ok(special_kl_div(args[0], args[1]))
        }
        SpecialCaseFunction::Ndtr => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("ndtr", mode));
            }
            special_scalar_from_tensor(special_ndtr(&special_scalar(args[0]), mode)?, "ndtr", mode)
        }
        SpecialCaseFunction::Ndtri => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("ndtri", mode));
            }
            special_scalar_from_tensor(
                special_ndtri(&special_scalar(args[0]), mode)?,
                "ndtri",
                mode,
            )
        }
        SpecialCaseFunction::Nrdtrimn => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("nrdtrimn", mode));
            }
            Ok(special_nrdtrimn(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Nrdtrisd => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("nrdtrisd", mode));
            }
            Ok(special_nrdtrisd(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::LogNdtr => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("log_ndtr", mode));
            }
            Ok(special_log_ndtr_scalar(args[0]))
        }
        SpecialCaseFunction::Logsumexp => {
            if args.is_empty() {
                return Err(special_invalid_fixture_error("logsumexp", mode));
            }
            Ok(special_logsumexp(&args))
        }
        SpecialCaseFunction::Log1p => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("log1p", mode));
            }
            Ok(special_log1p(args[0]))
        }
        SpecialCaseFunction::Expm1 => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("expm1", mode));
            }
            Ok(special_expm1(args[0]))
        }
        SpecialCaseFunction::Logaddexp => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("logaddexp", mode));
            }
            Ok(special_logaddexp(args[0], args[1]))
        }
        SpecialCaseFunction::Logaddexp2 => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("logaddexp2", mode));
            }
            Ok(special_logaddexp2(args[0], args[1]))
        }
        SpecialCaseFunction::Softplus => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("softplus", mode));
            }
            special_scalar_from_tensor(
                special_softplus(&special_scalar(args[0]), mode)?,
                "softplus",
                mode,
            )
        }
        SpecialCaseFunction::Huber => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("huber", mode));
            }
            special_scalar_from_tensor(
                special_huber(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "huber",
                mode,
            )
        }
        SpecialCaseFunction::PseudoHuber => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("pseudo_huber", mode));
            }
            special_scalar_from_tensor(
                special_pseudo_huber(&special_scalar(args[0]), &special_scalar(args[1]), mode)?,
                "pseudo_huber",
                mode,
            )
        }
        SpecialCaseFunction::Cosdg => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("cosdg", mode));
            }
            Ok(special_cosdg(args[0]))
        }
        SpecialCaseFunction::Sindg => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("sindg", mode));
            }
            Ok(special_sindg(args[0]))
        }
        SpecialCaseFunction::Tandg => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("tandg", mode));
            }
            Ok(special_tandg(args[0]))
        }
        SpecialCaseFunction::Cotdg => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("cotdg", mode));
            }
            Ok(special_cotdg(args[0]))
        }
        SpecialCaseFunction::Radian => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("radian", mode));
            }
            Ok(special_radian(args[0], args[1], args[2]))
        }
        SpecialCaseFunction::Cbrt => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("cbrt", mode));
            }
            Ok(special_cbrt(args[0]))
        }
        SpecialCaseFunction::Exp2 => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("exp2", mode));
            }
            Ok(special_exp2(args[0]))
        }
        SpecialCaseFunction::Exp10 => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("exp10", mode));
            }
            Ok(special_exp10(args[0]))
        }
        SpecialCaseFunction::Boxcox => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("boxcox", mode));
            }
            Ok(special_boxcox_transform(args[0], args[1]))
        }
        SpecialCaseFunction::InvBoxcox => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("inv_boxcox", mode));
            }
            Ok(special_inv_boxcox(args[0], args[1]))
        }
        SpecialCaseFunction::Boxcox1p => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("boxcox1p", mode));
            }
            Ok(special_boxcox1p(args[0], args[1]))
        }
        SpecialCaseFunction::InvBoxcox1p => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("inv_boxcox1p", mode));
            }
            Ok(special_inv_boxcox1p(args[0], args[1]))
        }
        SpecialCaseFunction::FresnelS => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("fresnel_s", mode));
            }
            let (s, _) = special_fresnel(args[0]);
            Ok(s)
        }
        SpecialCaseFunction::FresnelC => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("fresnel_c", mode));
            }
            let (_, c) = special_fresnel(args[0]);
            Ok(c)
        }
        SpecialCaseFunction::Dawsn => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("dawsn", mode));
            }
            special_scalar_from_tensor(
                special_dawsn(&special_scalar(args[0]), mode)?,
                "dawsn",
                mode,
            )
        }
        SpecialCaseFunction::SiciSi => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("sici_si", mode));
            }
            let (si, _) = special_sici(args[0]);
            Ok(si)
        }
        SpecialCaseFunction::SiciCi => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("sici_ci", mode));
            }
            let (_, ci) = special_sici(args[0]);
            Ok(ci)
        }
        SpecialCaseFunction::ShichiShi => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("shichi_shi", mode));
            }
            let (shi, _) = special_shichi(args[0]);
            Ok(shi)
        }
        SpecialCaseFunction::ShichiChi => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("shichi_chi", mode));
            }
            let (_, chi) = special_shichi(args[0]);
            Ok(chi)
        }
        SpecialCaseFunction::Struve => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("struve", mode));
            }
            Ok(special_struve(args[0], args[1]))
        }
        SpecialCaseFunction::Modstruve => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("modstruve", mode));
            }
            Ok(special_modstruve(args[0], args[1]))
        }
        SpecialCaseFunction::Ber => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("ber", mode));
            }
            Ok(special_ber(args[0]))
        }
        SpecialCaseFunction::Bei => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("bei", mode));
            }
            Ok(special_bei(args[0]))
        }
        SpecialCaseFunction::Ker => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("ker", mode));
            }
            Ok(special_ker(args[0]))
        }
        SpecialCaseFunction::Kei => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("kei", mode));
            }
            Ok(special_kei(args[0]))
        }
        SpecialCaseFunction::Spence => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("spence", mode));
            }
            special_scalar_from_tensor(
                special_spence(&special_scalar(args[0]), mode)?,
                "spence",
                mode,
            )
        }
        SpecialCaseFunction::Zeta => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("zeta", mode));
            }
            Ok(special_zeta(args[0]))
        }
        SpecialCaseFunction::Zetac => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("zetac", mode));
            }
            Ok(special_zetac(args[0]))
        }
        SpecialCaseFunction::HurwitzZeta => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("hurwitz_zeta", mode));
            }
            Ok(special_hurwitz_zeta(args[0], args[1]))
        }
        SpecialCaseFunction::Lpmv => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("lpmv", mode));
            }
            let order = special_i32_from_fixture("lpmv", mode, args[0])?;
            let degree = special_u32_from_fixture("lpmv", mode, args[1])?;
            Ok(special_lpmv(order, degree, args[2]))
        }
        SpecialCaseFunction::SphHarmY => Err(special_invalid_fixture_error("sph_harm_y", mode)),
        SpecialCaseFunction::EvalLegendre => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("eval_legendre", mode));
            }
            let degree = special_u32_from_fixture("eval_legendre", mode, args[0])?;
            Ok(special_eval_legendre(degree, args[1]))
        }
        SpecialCaseFunction::EvalChebyt => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("eval_chebyt", mode));
            }
            let degree = special_u32_from_fixture("eval_chebyt", mode, args[0])?;
            Ok(special_eval_chebyt(degree, args[1]))
        }
        SpecialCaseFunction::EvalChebyu => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("eval_chebyu", mode));
            }
            let degree = special_u32_from_fixture("eval_chebyu", mode, args[0])?;
            Ok(special_eval_chebyu(degree, args[1]))
        }
        SpecialCaseFunction::EvalHermite => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("eval_hermite", mode));
            }
            let degree = special_u32_from_fixture("eval_hermite", mode, args[0])?;
            Ok(special_eval_hermite(degree, args[1]))
        }
        SpecialCaseFunction::EvalHermitenorm => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("eval_hermitenorm", mode));
            }
            let degree = special_u32_from_fixture("eval_hermitenorm", mode, args[0])?;
            Ok(special_eval_hermitenorm(degree, args[1]))
        }
        SpecialCaseFunction::EvalLaguerre => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("eval_laguerre", mode));
            }
            let degree = special_u32_from_fixture("eval_laguerre", mode, args[0])?;
            Ok(special_eval_laguerre(degree, args[1]))
        }
        SpecialCaseFunction::EvalGenlaguerre => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("eval_genlaguerre", mode));
            }
            let degree = special_u32_from_fixture("eval_genlaguerre", mode, args[0])?;
            Ok(special_eval_genlaguerre(degree, args[1], args[2]))
        }
        SpecialCaseFunction::EvalJacobi => {
            if args.len() != 4 {
                return Err(special_invalid_fixture_error("eval_jacobi", mode));
            }
            let degree = special_u32_from_fixture("eval_jacobi", mode, args[0])?;
            Ok(special_eval_jacobi(degree, args[1], args[2], args[3]))
        }
        SpecialCaseFunction::EvalGegenbauer => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("eval_gegenbauer", mode));
            }
            let degree = special_u32_from_fixture("eval_gegenbauer", mode, args[0])?;
            Ok(special_eval_gegenbauer(degree, args[1], args[2]))
        }
        SpecialCaseFunction::EvalShLegendre => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("eval_sh_legendre", mode));
            }
            let degree = special_u32_from_fixture("eval_sh_legendre", mode, args[0])?;
            Ok(special_eval_sh_legendre(degree, args[1]))
        }
        SpecialCaseFunction::EvalShChebyt => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("eval_sh_chebyt", mode));
            }
            let degree = special_u32_from_fixture("eval_sh_chebyt", mode, args[0])?;
            Ok(special_eval_sh_chebyt(degree, args[1]))
        }
        SpecialCaseFunction::EvalShChebyu => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("eval_sh_chebyu", mode));
            }
            let degree = special_u32_from_fixture("eval_sh_chebyu", mode, args[0])?;
            Ok(special_eval_sh_chebyu(degree, args[1]))
        }
        SpecialCaseFunction::RootsChebytNode => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("roots_chebyt_node", mode));
            }
            let degree = special_usize_from_fixture("roots_chebyt_node", mode, args[0])?;
            let index = special_usize_from_fixture("roots_chebyt_node", mode, args[1])?;
            let (nodes, _) = special_roots_chebyt(degree);
            roots_component(nodes.get(index), "roots_chebyt_node", mode)
        }
        SpecialCaseFunction::RootsChebytWeight => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("roots_chebyt_weight", mode));
            }
            let degree = special_usize_from_fixture("roots_chebyt_weight", mode, args[0])?;
            let index = special_usize_from_fixture("roots_chebyt_weight", mode, args[1])?;
            let (_, weights) = special_roots_chebyt(degree);
            roots_component(weights.get(index), "roots_chebyt_weight", mode)
        }
        SpecialCaseFunction::RootsChebyuNode => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("roots_chebyu_node", mode));
            }
            let degree = special_usize_from_fixture("roots_chebyu_node", mode, args[0])?;
            let index = special_usize_from_fixture("roots_chebyu_node", mode, args[1])?;
            let (nodes, _) = special_roots_chebyu(degree);
            roots_component(nodes.get(index), "roots_chebyu_node", mode)
        }
        SpecialCaseFunction::RootsChebyuWeight => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("roots_chebyu_weight", mode));
            }
            let degree = special_usize_from_fixture("roots_chebyu_weight", mode, args[0])?;
            let index = special_usize_from_fixture("roots_chebyu_weight", mode, args[1])?;
            let (_, weights) = special_roots_chebyu(degree);
            roots_component(weights.get(index), "roots_chebyu_weight", mode)
        }
        SpecialCaseFunction::RootsHermiteNode => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("roots_hermite_node", mode));
            }
            let degree = special_usize_from_fixture("roots_hermite_node", mode, args[0])?;
            let index = special_usize_from_fixture("roots_hermite_node", mode, args[1])?;
            let (nodes, _) = special_roots_hermite(degree);
            roots_component(nodes.get(index), "roots_hermite_node", mode)
        }
        SpecialCaseFunction::RootsHermiteWeight => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("roots_hermite_weight", mode));
            }
            let degree = special_usize_from_fixture("roots_hermite_weight", mode, args[0])?;
            let index = special_usize_from_fixture("roots_hermite_weight", mode, args[1])?;
            let (_, weights) = special_roots_hermite(degree);
            roots_component(weights.get(index), "roots_hermite_weight", mode)
        }
        SpecialCaseFunction::RootsHermitenormNode => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error(
                    "roots_hermitenorm_node",
                    mode,
                ));
            }
            let degree = special_usize_from_fixture("roots_hermitenorm_node", mode, args[0])?;
            let index = special_usize_from_fixture("roots_hermitenorm_node", mode, args[1])?;
            let (nodes, _) = special_roots_hermitenorm(degree);
            roots_component(nodes.get(index), "roots_hermitenorm_node", mode)
        }
        SpecialCaseFunction::RootsHermitenormWeight => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error(
                    "roots_hermitenorm_weight",
                    mode,
                ));
            }
            let degree = special_usize_from_fixture("roots_hermitenorm_weight", mode, args[0])?;
            let index = special_usize_from_fixture("roots_hermitenorm_weight", mode, args[1])?;
            let (_, weights) = special_roots_hermitenorm(degree);
            roots_component(weights.get(index), "roots_hermitenorm_weight", mode)
        }
        SpecialCaseFunction::RootsLaguerreNode => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("roots_laguerre_node", mode));
            }
            let degree = special_usize_from_fixture("roots_laguerre_node", mode, args[0])?;
            let index = special_usize_from_fixture("roots_laguerre_node", mode, args[1])?;
            let (nodes, _) = special_roots_laguerre(degree);
            roots_component(nodes.get(index), "roots_laguerre_node", mode)
        }
        SpecialCaseFunction::RootsLaguerreWeight => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("roots_laguerre_weight", mode));
            }
            let degree = special_usize_from_fixture("roots_laguerre_weight", mode, args[0])?;
            let index = special_usize_from_fixture("roots_laguerre_weight", mode, args[1])?;
            let (_, weights) = special_roots_laguerre(degree);
            roots_component(weights.get(index), "roots_laguerre_weight", mode)
        }
        SpecialCaseFunction::RootsGenlaguerreNode => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error(
                    "roots_genlaguerre_node",
                    mode,
                ));
            }
            let degree = special_usize_from_fixture("roots_genlaguerre_node", mode, args[0])?;
            let index = special_usize_from_fixture("roots_genlaguerre_node", mode, args[2])?;
            let (nodes, _) = special_roots_genlaguerre(degree, args[1]);
            roots_component(nodes.get(index), "roots_genlaguerre_node", mode)
        }
        SpecialCaseFunction::RootsGenlaguerreWeight => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error(
                    "roots_genlaguerre_weight",
                    mode,
                ));
            }
            let degree = special_usize_from_fixture("roots_genlaguerre_weight", mode, args[0])?;
            let index = special_usize_from_fixture("roots_genlaguerre_weight", mode, args[2])?;
            let (_, weights) = special_roots_genlaguerre(degree, args[1]);
            roots_component(weights.get(index), "roots_genlaguerre_weight", mode)
        }
        SpecialCaseFunction::RootsJacobiNode => {
            if args.len() != 4 {
                return Err(special_invalid_fixture_error("roots_jacobi_node", mode));
            }
            let degree = special_usize_from_fixture("roots_jacobi_node", mode, args[0])?;
            let index = special_usize_from_fixture("roots_jacobi_node", mode, args[3])?;
            let (nodes, _) = special_roots_jacobi(degree, args[1], args[2]);
            roots_component(nodes.get(index), "roots_jacobi_node", mode)
        }
        SpecialCaseFunction::RootsJacobiWeight => {
            if args.len() != 4 {
                return Err(special_invalid_fixture_error("roots_jacobi_weight", mode));
            }
            let degree = special_usize_from_fixture("roots_jacobi_weight", mode, args[0])?;
            let index = special_usize_from_fixture("roots_jacobi_weight", mode, args[3])?;
            let (_, weights) = special_roots_jacobi(degree, args[1], args[2]);
            roots_component(weights.get(index), "roots_jacobi_weight", mode)
        }
        SpecialCaseFunction::RootsGegenbauerNode => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("roots_gegenbauer_node", mode));
            }
            let degree = special_usize_from_fixture("roots_gegenbauer_node", mode, args[0])?;
            let index = special_usize_from_fixture("roots_gegenbauer_node", mode, args[2])?;
            let (nodes, _) = special_roots_gegenbauer(degree, args[1]);
            roots_component(nodes.get(index), "roots_gegenbauer_node", mode)
        }
        SpecialCaseFunction::RootsGegenbauerWeight => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error(
                    "roots_gegenbauer_weight",
                    mode,
                ));
            }
            let degree = special_usize_from_fixture("roots_gegenbauer_weight", mode, args[0])?;
            let index = special_usize_from_fixture("roots_gegenbauer_weight", mode, args[2])?;
            let (_, weights) = special_roots_gegenbauer(degree, args[1]);
            roots_component(weights.get(index), "roots_gegenbauer_weight", mode)
        }
        SpecialCaseFunction::RootsLegendreNode => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("roots_legendre_node", mode));
            }
            let degree = special_usize_from_fixture("roots_legendre_node", mode, args[0])?;
            let index = special_usize_from_fixture("roots_legendre_node", mode, args[1])?;
            let (nodes, _) = special_roots_legendre(degree);
            roots_component(nodes.get(index), "roots_legendre_node", mode)
        }
        SpecialCaseFunction::RootsLegendreWeight => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("roots_legendre_weight", mode));
            }
            let degree = special_usize_from_fixture("roots_legendre_weight", mode, args[0])?;
            let index = special_usize_from_fixture("roots_legendre_weight", mode, args[1])?;
            let (_, weights) = special_roots_legendre(degree);
            roots_component(weights.get(index), "roots_legendre_weight", mode)
        }
        SpecialCaseFunction::Expit => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("expit", mode));
            }
            special_scalar_from_tensor(
                special_expit(&special_scalar(args[0]), mode)?,
                "expit",
                mode,
            )
        }
        SpecialCaseFunction::Logit => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("logit", mode));
            }
            special_scalar_from_tensor(
                special_logit(&special_scalar(args[0]), mode)?,
                "logit",
                mode,
            )
        }
        SpecialCaseFunction::Exprel => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("exprel", mode));
            }
            special_scalar_from_tensor(
                special_exprel(&special_scalar(args[0]), mode)?,
                "exprel",
                mode,
            )
        }
        SpecialCaseFunction::RelErfErfcIdentity => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("rel_erf_erfc_identity", mode));
            }
            let x = args[0];
            let erf_v =
                special_scalar_from_tensor(special_erf(&special_scalar(x), mode)?, "erf", mode)?;
            let erfc_v =
                special_scalar_from_tensor(special_erfc(&special_scalar(x), mode)?, "erfc", mode)?;
            Ok(erf_v + erfc_v)
        }
        SpecialCaseFunction::RelGammaRecurrence => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("rel_gamma_recurrence", mode));
            }
            let x = args[0];
            let gx = special_scalar_from_tensor(
                special_gamma(&special_scalar(x), mode)?,
                "gamma",
                mode,
            )?;
            let gx1 = special_scalar_from_tensor(
                special_gamma(&special_scalar(x + 1.0), mode)?,
                "gamma",
                mode,
            )?;
            Ok(gx1 - x * gx)
        }
        SpecialCaseFunction::RelBetaSymmetry => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("rel_beta_symmetry", mode));
            }
            let a = args[0];
            let b = args[1];
            let lhs = special_scalar_from_tensor(
                special_beta(&special_scalar(a), &special_scalar(b), mode)?,
                "beta",
                mode,
            )?;
            let rhs = special_scalar_from_tensor(
                special_beta(&special_scalar(b), &special_scalar(a), mode)?,
                "beta",
                mode,
            )?;
            Ok(lhs - rhs)
        }
        SpecialCaseFunction::RelGammaincComplement => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error(
                    "rel_gammainc_complement",
                    mode,
                ));
            }
            let a = args[0];
            let x = args[1];
            let p = special_scalar_from_tensor(
                special_gammainc(&special_scalar(a), &special_scalar(x), mode)?,
                "gammainc",
                mode,
            )?;
            let q = special_scalar_from_tensor(
                special_gammaincc(&special_scalar(a), &special_scalar(x), mode)?,
                "gammaincc",
                mode,
            )?;
            Ok(p + q)
        }
        SpecialCaseFunction::RelJnRecurrence => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("rel_jn_recurrence", mode));
            }
            let n = args[0];
            let x = args[1];
            let jn_prev = special_scalar_from_tensor(
                special_jn(&special_scalar(n - 1.0), &special_scalar(x), mode)?,
                "jn",
                mode,
            )?;
            let jn_curr = special_scalar_from_tensor(
                special_jn(&special_scalar(n), &special_scalar(x), mode)?,
                "jn",
                mode,
            )?;
            let jn_next = special_scalar_from_tensor(
                special_jn(&special_scalar(n + 1.0), &special_scalar(x), mode)?,
                "jn",
                mode,
            )?;
            Ok(jn_next - ((2.0 * n / x) * jn_curr - jn_prev))
        }
        SpecialCaseFunction::RelErfinvComposition => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error(
                    "rel_erfinv_composition",
                    mode,
                ));
            }
            let x = args[0];
            let erf_x =
                special_scalar_from_tensor(special_erf(&special_scalar(x), mode)?, "erf", mode)?;
            special_scalar_from_tensor(
                special_erfinv(&special_scalar(erf_x), mode)?,
                "erfinv",
                mode,
            )
        }
    }
}

fn execute_special_case(case: &SpecialCase) -> Result<SpecialObservedOutcome, FsciSpecialError> {
    let mode = case.mode;
    let args = &case.args;
    match case.function {
        SpecialCaseFunction::Beta => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("beta", mode));
            }
            Ok(special_observed_from_tensor(special_beta(
                &special_tensor_from_argument(&args[0]),
                &special_tensor_from_argument(&args[1]),
                mode,
            )?))
        }
        SpecialCaseFunction::Betaln => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("betaln", mode));
            }
            Ok(special_observed_from_tensor(special_betaln(
                &special_tensor_from_argument(&args[0]),
                &special_tensor_from_argument(&args[1]),
                mode,
            )?))
        }
        SpecialCaseFunction::Betainc => {
            if args.len() != 3 {
                return Err(special_invalid_fixture_error("betainc", mode));
            }
            Ok(special_observed_from_tensor(special_betainc(
                &special_tensor_from_argument(&args[0]),
                &special_tensor_from_argument(&args[1]),
                &special_tensor_from_argument(&args[2]),
                mode,
            )?))
        }
        SpecialCaseFunction::Ellipk => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("ellipk", mode));
            }
            Ok(special_observed_from_tensor(special_ellipk(
                &special_tensor_from_argument(&args[0]),
                mode,
            )?))
        }
        SpecialCaseFunction::Ellipkm1 => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("ellipkm1", mode));
            }
            Ok(special_observed_from_tensor(special_ellipkm1(
                &special_tensor_from_argument(&args[0]),
                mode,
            )?))
        }
        SpecialCaseFunction::Ellipe => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("ellipe", mode));
            }
            Ok(special_observed_from_tensor(special_ellipe(
                &special_tensor_from_argument(&args[0]),
                mode,
            )?))
        }
        SpecialCaseFunction::Ellipkinc => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("ellipkinc", mode));
            }
            Ok(special_observed_from_tensor(special_ellipkinc(
                &special_tensor_from_argument(&args[0]),
                &special_tensor_from_argument(&args[1]),
                mode,
            )?))
        }
        SpecialCaseFunction::Ellipeinc => {
            if args.len() != 2 {
                return Err(special_invalid_fixture_error("ellipeinc", mode));
            }
            Ok(special_observed_from_tensor(special_ellipeinc(
                &special_tensor_from_argument(&args[0]),
                &special_tensor_from_argument(&args[1]),
                mode,
            )?))
        }
        SpecialCaseFunction::Lambertw => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("lambertw", mode));
            }
            Ok(special_observed_from_tensor(special_lambertw(
                &special_tensor_from_argument(&args[0]),
                mode,
            )?))
        }
        SpecialCaseFunction::Exp1 => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("exp1", mode));
            }
            Ok(special_observed_from_tensor(special_exp1(
                &special_tensor_from_argument(&args[0]),
                mode,
            )?))
        }
        SpecialCaseFunction::Expi => {
            if args.len() != 1 {
                return Err(special_invalid_fixture_error("expi", mode));
            }
            Ok(special_observed_from_tensor(special_expi(
                &special_tensor_from_argument(&args[0]),
                mode,
            )?))
        }
        SpecialCaseFunction::Jvp => {
            execute_special_bessel_derivative_case("jvp", args, mode, special_jvp)
        }
        SpecialCaseFunction::Yvp => {
            execute_special_bessel_derivative_case("yvp", args, mode, special_yvp)
        }
        SpecialCaseFunction::Ivp => {
            execute_special_bessel_derivative_case("ivp", args, mode, special_ivp)
        }
        SpecialCaseFunction::Kvp => {
            execute_special_bessel_derivative_case("kvp", args, mode, special_kvp)
        }
        SpecialCaseFunction::SphHarmY => {
            let values = case
                .real_scalar_args()
                .ok_or_else(|| special_invalid_fixture_error("sph_harm_y", mode))?;
            if values.len() != 4 {
                return Err(special_invalid_fixture_error("sph_harm_y", mode));
            }
            let degree = special_u32_from_fixture("sph_harm_y", mode, values[0])?;
            let order = special_i32_from_fixture("sph_harm_y", mode, values[1])?;
            let value = special_sph_harm_y(degree, order, values[2], values[3]);
            Ok(SpecialObservedOutcome::ComplexScalar([value.re, value.im]))
        }
        _ => execute_special_case_scalar(case).map(SpecialObservedOutcome::Scalar),
    }
}

fn special_scalar(value: f64) -> FsciSpecialTensor {
    FsciSpecialTensor::RealScalar(value)
}

fn special_complex_value_to_complex64(value: &SpecialComplexValue) -> Complex64 {
    Complex64::new(value.re, value.im)
}

fn special_tensor_from_argument(arg: &SpecialArgument) -> FsciSpecialTensor {
    match arg {
        SpecialArgument::RealScalar(value) => FsciSpecialTensor::RealScalar(*value),
        SpecialArgument::ComplexScalar(value) => {
            FsciSpecialTensor::ComplexScalar(special_complex_value_to_complex64(value))
        }
        SpecialArgument::RealVector { values } => FsciSpecialTensor::RealVec(values.clone()),
        SpecialArgument::ComplexVector { values } => FsciSpecialTensor::ComplexVec(
            values
                .iter()
                .map(special_complex_value_to_complex64)
                .collect(),
        ),
    }
}

fn special_observed_from_tensor(tensor: FsciSpecialTensor) -> SpecialObservedOutcome {
    match tensor {
        FsciSpecialTensor::Empty => SpecialObservedOutcome::Vector(Vec::new()),
        FsciSpecialTensor::RealScalar(value) => SpecialObservedOutcome::Scalar(value),
        FsciSpecialTensor::RealVec(values) => SpecialObservedOutcome::Vector(values),
        FsciSpecialTensor::ComplexScalar(value) => {
            SpecialObservedOutcome::ComplexScalar([value.re, value.im])
        }
        FsciSpecialTensor::ComplexVec(values) => SpecialObservedOutcome::ComplexVector(
            values.into_iter().map(|v| [v.re, v.im]).collect(),
        ),
    }
}

fn execute_special_bessel_derivative_case(
    function: &'static str,
    args: &[SpecialArgument],
    mode: RuntimeMode,
    kernel: fn(
        &FsciSpecialTensor,
        &FsciSpecialTensor,
        usize,
        RuntimeMode,
    ) -> Result<FsciSpecialTensor, FsciSpecialError>,
) -> Result<SpecialObservedOutcome, FsciSpecialError> {
    if args.len() != 3 {
        return Err(special_invalid_fixture_error(function, mode));
    }
    let derivative_order =
        special_derivative_order_argument_from_fixture(function, mode, &args[2])?;
    Ok(special_observed_from_tensor(kernel(
        &special_tensor_from_argument(&args[0]),
        &special_tensor_from_argument(&args[1]),
        derivative_order,
        mode,
    )?))
}

fn special_derivative_order_argument_from_fixture(
    function: &'static str,
    mode: RuntimeMode,
    arg: &SpecialArgument,
) -> Result<usize, FsciSpecialError> {
    match arg {
        SpecialArgument::RealScalar(value) => {
            special_derivative_order_from_fixture(function, mode, *value)
        }
        _ => Err(FsciSpecialError {
            function,
            kind: FsciSpecialErrorKind::FixtureSchemaError,
            mode,
            detail: "derivative order fixture must be a real scalar",
        }),
    }
}

fn special_derivative_order_from_fixture(
    function: &'static str,
    mode: RuntimeMode,
    value: f64,
) -> Result<usize, FsciSpecialError> {
    if value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= usize::MAX as f64 {
        return Ok(value as usize);
    }
    Err(FsciSpecialError {
        function,
        kind: FsciSpecialErrorKind::FixtureSchemaError,
        mode,
        detail: "derivative order fixture must be a non-negative integer",
    })
}

fn special_i32_from_fixture(
    function: &'static str,
    mode: RuntimeMode,
    value: f64,
) -> Result<i32, FsciSpecialError> {
    if value.is_finite()
        && value.fract() == 0.0
        && value >= f64::from(i32::MIN)
        && value <= f64::from(i32::MAX)
    {
        return Ok(value as i32);
    }
    Err(FsciSpecialError {
        function,
        kind: FsciSpecialErrorKind::FixtureSchemaError,
        mode,
        detail: "fixture argument must be a finite i32 integer",
    })
}

fn special_i64_from_fixture(
    function: &'static str,
    mode: RuntimeMode,
    value: f64,
) -> Result<i64, FsciSpecialError> {
    if value.is_finite()
        && value.fract() == 0.0
        && value >= i64::MIN as f64
        && value <= i64::MAX as f64
    {
        return Ok(value as i64);
    }
    Err(FsciSpecialError {
        function,
        kind: FsciSpecialErrorKind::FixtureSchemaError,
        mode,
        detail: "fixture argument must be a finite i64 integer",
    })
}

fn special_u32_from_fixture(
    function: &'static str,
    mode: RuntimeMode,
    value: f64,
) -> Result<u32, FsciSpecialError> {
    if value.is_finite() && value.fract() == 0.0 && value >= 0.0 && value <= f64::from(u32::MAX) {
        return Ok(value as u32);
    }
    Err(FsciSpecialError {
        function,
        kind: FsciSpecialErrorKind::FixtureSchemaError,
        mode,
        detail: "fixture argument must be a finite u32 integer",
    })
}

fn special_u64_from_fixture(
    function: &'static str,
    mode: RuntimeMode,
    value: f64,
) -> Result<u64, FsciSpecialError> {
    if value.is_finite() && value.fract() == 0.0 && value >= 0.0 && value <= u64::MAX as f64 {
        return Ok(value as u64);
    }
    Err(FsciSpecialError {
        function,
        kind: FsciSpecialErrorKind::FixtureSchemaError,
        mode,
        detail: "fixture argument must be a finite u64 integer",
    })
}

fn special_usize_from_fixture(
    function: &'static str,
    mode: RuntimeMode,
    value: f64,
) -> Result<usize, FsciSpecialError> {
    if value.is_finite() && value.fract() == 0.0 && value >= 0.0 && value <= usize::MAX as f64 {
        return Ok(value as usize);
    }
    Err(FsciSpecialError {
        function,
        kind: FsciSpecialErrorKind::FixtureSchemaError,
        mode,
        detail: "fixture argument must be a finite usize integer",
    })
}

fn roots_component(
    component: Option<&f64>,
    function: &'static str,
    mode: RuntimeMode,
) -> Result<f64, FsciSpecialError> {
    component.copied().ok_or(FsciSpecialError {
        function,
        kind: FsciSpecialErrorKind::FixtureSchemaError,
        mode,
        detail: "roots fixture index is out of range",
    })
}

fn special_scalar_from_tensor(
    tensor: FsciSpecialTensor,
    function: &'static str,
    mode: RuntimeMode,
) -> Result<f64, FsciSpecialError> {
    match tensor {
        FsciSpecialTensor::RealScalar(value) => Ok(value),
        _ => Err(FsciSpecialError {
            function,
            kind: FsciSpecialErrorKind::FixtureSchemaError,
            mode,
            detail: "expected scalar output in conformance fixture",
        }),
    }
}

fn special_invalid_fixture_error(function: &'static str, mode: RuntimeMode) -> FsciSpecialError {
    FsciSpecialError {
        function,
        kind: FsciSpecialErrorKind::FixtureSchemaError,
        mode,
        detail: "invalid fixture arity for special case",
    }
}

fn compare_special_case_differential(
    case: &SpecialCase,
    observed: &Result<SpecialObservedOutcome, FsciSpecialError>,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    match (&case.expected, observed) {
        (
            SpecialExpectedOutcome::Scalar {
                value,
                atol,
                rtol,
                contract_ref,
            },
            Ok(SpecialObservedOutcome::Scalar(actual)),
        ) => {
            let tolerance =
                resolve_special_contract_tolerance(contract_ref.as_deref(), *atol, *rtol);
            let diff = (actual - *value).abs();
            let pass = allclose_scalar(*actual, *value, tolerance.atol, tolerance.rtol);
            let msg = if pass {
                format!("special scalar matched (diff={diff:.2e})")
            } else {
                format!(
                    "special scalar mismatch: expected={value:.16e}, got={actual:.16e}, diff={diff:.2e}, atol={:.2e}, rtol={:.2e}",
                    tolerance.atol, tolerance.rtol
                )
            };
            (pass, msg, Some(diff), Some(tolerance))
        }
        (
            SpecialExpectedOutcome::Vector {
                values,
                atol,
                rtol,
                contract_ref,
            },
            Ok(SpecialObservedOutcome::Vector(actual)),
        ) => {
            let tolerance =
                resolve_special_contract_tolerance(contract_ref.as_deref(), *atol, *rtol);
            if values.len() != actual.len() {
                return (
                    false,
                    format!(
                        "special vector length mismatch: expected={}, got={}",
                        values.len(),
                        actual.len()
                    ),
                    None,
                    Some(tolerance),
                );
            }
            let diff = values
                .iter()
                .zip(actual.iter())
                .map(|(expected, got)| (got - expected).abs())
                .fold(0.0, f64::max);
            let pass = values.iter().zip(actual.iter()).all(|(expected, got)| {
                allclose_scalar(*got, *expected, tolerance.atol, tolerance.rtol)
            });
            let msg = if pass {
                format!("special vector matched (max_diff={diff:.2e})")
            } else {
                format!(
                    "special vector mismatch: expected={values:?}, got={actual:?}, max_diff={diff:.2e}, atol={:.2e}, rtol={:.2e}",
                    tolerance.atol, tolerance.rtol
                )
            };
            (pass, msg, Some(diff), Some(tolerance))
        }
        (
            SpecialExpectedOutcome::ComplexScalar {
                value,
                atol,
                rtol,
                contract_ref,
            },
            Ok(SpecialObservedOutcome::ComplexScalar(actual)),
        ) => {
            let tolerance =
                resolve_special_contract_tolerance(contract_ref.as_deref(), *atol, *rtol);
            let expected = [value.re, value.im];
            let diff = special_componentwise_max_diff(*actual, expected);
            let pass = special_complex_allclose(*actual, expected, tolerance.atol, tolerance.rtol);
            let msg = if pass {
                format!("special complex scalar matched (max_diff={diff:.2e})")
            } else {
                format!(
                    "special complex scalar mismatch: expected={expected:?}, got={actual:?}, max_diff={diff:.2e}, atol={:.2e}, rtol={:.2e}",
                    tolerance.atol, tolerance.rtol
                )
            };
            (pass, msg, Some(diff), Some(tolerance))
        }
        (
            SpecialExpectedOutcome::ComplexVector {
                values,
                atol,
                rtol,
                contract_ref,
            },
            Ok(SpecialObservedOutcome::ComplexVector(actual)),
        ) => {
            let tolerance =
                resolve_special_contract_tolerance(contract_ref.as_deref(), *atol, *rtol);
            if values.len() != actual.len() {
                return (
                    false,
                    format!(
                        "special complex vector length mismatch: expected={}, got={}",
                        values.len(),
                        actual.len()
                    ),
                    None,
                    Some(tolerance),
                );
            }
            let expected: Vec<[f64; 2]> = values.iter().map(|value| [value.re, value.im]).collect();
            let diff = expected
                .iter()
                .zip(actual.iter())
                .map(|(expected, got)| special_componentwise_max_diff(*got, *expected))
                .fold(0.0, f64::max);
            let pass = expected.iter().zip(actual.iter()).all(|(expected, got)| {
                special_complex_allclose(*got, *expected, tolerance.atol, tolerance.rtol)
            });
            let msg = if pass {
                format!("special complex vector matched (max_diff={diff:.2e})")
            } else {
                format!(
                    "special complex vector mismatch: expected={expected:?}, got={actual:?}, max_diff={diff:.2e}, atol={:.2e}, rtol={:.2e}",
                    tolerance.atol, tolerance.rtol
                )
            };
            (pass, msg, Some(diff), Some(tolerance))
        }
        (SpecialExpectedOutcome::Class { class }, Ok(actual)) => {
            let observed_class = classify_special_observed(actual);
            let pass = observed_class == *class;
            let msg = if pass {
                format!("value class matched ({observed_class:?})")
            } else {
                format!("value class mismatch: expected={class:?}, got={observed_class:?}")
            };
            (pass, msg, None, None)
        }
        (SpecialExpectedOutcome::ErrorKind { error }, Err(actual)) => {
            // Previously compared format!("{:?}", actual.kind) == *error as
            // strings. That broke when the Debug repr changed (e.g.,
            // variants growing fields), and silently accepted any typo in
            // the fixture. Replaced with typed enum comparison via
            // parse_special_error_kind; unknown fixture strings are
            // surfaced as a judgeable mismatch. See frankenscipy-yxml.
            let Some(expected_kind) = parse_special_error_kind(error) else {
                return (
                    false,
                    format!(
                        "unknown error kind `{error}` in fixture (expected one of \
                         DomainError, FixtureSchemaError, PoleInput, NonFiniteInput, \
                         CancellationRisk, OverflowRisk, SingularityRisk, \
                         NotYetImplemented, ShapeMismatch)"
                    ),
                    None,
                    None,
                );
            };
            let pass = actual.kind == expected_kind;
            let observed_kind = format!("{:?}", actual.kind);
            let msg = if pass {
                format!("error kind matched ({observed_kind})")
            } else {
                format!("error kind mismatch: expected={error}, got={observed_kind}")
            };
            (pass, msg, None, None)
        }
        (expected, observed) => (
            false,
            format!("shape mismatch: expected {expected:?}, got {observed:?}"),
            None,
            None,
        ),
    }
}

fn special_componentwise_max_diff(actual: [f64; 2], expected: [f64; 2]) -> f64 {
    (actual[0] - expected[0])
        .abs()
        .max((actual[1] - expected[1]).abs())
}

fn special_complex_allclose(actual: [f64; 2], expected: [f64; 2], atol: f64, rtol: f64) -> bool {
    allclose_scalar(actual[0], expected[0], atol, rtol)
        && allclose_scalar(actual[1], expected[1], atol, rtol)
}

fn classify_special_observed(observed: &SpecialObservedOutcome) -> SpecialValueClass {
    match observed {
        SpecialObservedOutcome::Scalar(value) => classify_special_value(*value),
        SpecialObservedOutcome::Vector(values) => {
            classify_special_sequence(values.iter().copied().map(classify_special_value))
        }
        SpecialObservedOutcome::ComplexScalar(value) => {
            classify_special_sequence(value.iter().copied().map(classify_special_value))
        }
        SpecialObservedOutcome::ComplexVector(values) => classify_special_sequence(
            values
                .iter()
                .flat_map(|value| value.iter().copied())
                .map(classify_special_value),
        ),
    }
}

fn classify_special_sequence<I>(classes: I) -> SpecialValueClass
where
    I: IntoIterator<Item = SpecialValueClass>,
{
    let mut saw_pos_inf = false;
    let mut saw_neg_inf = false;
    for class in classes {
        match class {
            SpecialValueClass::Nan => return SpecialValueClass::Nan,
            SpecialValueClass::PosInf => saw_pos_inf = true,
            SpecialValueClass::NegInf => saw_neg_inf = true,
            SpecialValueClass::Finite => {}
        }
    }
    if saw_pos_inf {
        SpecialValueClass::PosInf
    } else if saw_neg_inf {
        SpecialValueClass::NegInf
    } else {
        SpecialValueClass::Finite
    }
}

fn classify_special_value(value: f64) -> SpecialValueClass {
    if value.is_nan() {
        return SpecialValueClass::Nan;
    }
    if value == f64::INFINITY {
        return SpecialValueClass::PosInf;
    }
    if value == f64::NEG_INFINITY {
        return SpecialValueClass::NegInf;
    }
    SpecialValueClass::Finite
}

/// Compare a linalg case and return (passed, message, max_diff, tolerance_used).
fn compare_linalg_case_differential(
    expected: &LinalgExpectedOutcome,
    observed: &Result<LinalgObservedOutcome, LinalgError>,
) -> (bool, String, Option<f64>, Option<ToleranceUsed>) {
    match (expected, observed) {
        (
            LinalgExpectedOutcome::Vector {
                values,
                atol,
                rtol,
                expect_warning_ill_conditioned,
            },
            Ok(LinalgObservedOutcome::Vector {
                values: got,
                warning_ill_conditioned,
            }),
        ) => {
            let md = max_diff_vec(got, values);
            let vectors_match = allclose_vec(got, values, *atol, *rtol);
            let warning_match = expect_warning_ill_conditioned
                .is_none_or(|expect| expect == *warning_ill_conditioned);
            let pass = vectors_match && warning_match;
            let msg = if pass {
                format!("vector matched (max_diff={md:.2e})")
            } else {
                format!("vector mismatch: max_diff={md:.2e}, atol={atol}, rtol={rtol}")
            };
            (
                pass,
                msg,
                Some(md),
                Some(ToleranceUsed {
                    atol: *atol,
                    rtol: *rtol,
                    comparison_mode: "mixed".to_owned(),
                }),
            )
        }
        (
            LinalgExpectedOutcome::Matrix { values, atol, rtol },
            Ok(LinalgObservedOutcome::Matrix { values: got }),
        ) => {
            let md = max_diff_matrix(got, values);
            let pass = allclose_matrix(got, values, *atol, *rtol);
            let msg = if pass {
                format!("matrix matched (max_diff={md:.2e})")
            } else {
                format!("matrix mismatch: max_diff={md:.2e}, atol={atol}, rtol={rtol}")
            };
            (
                pass,
                msg,
                Some(md),
                Some(ToleranceUsed {
                    atol: *atol,
                    rtol: *rtol,
                    comparison_mode: "mixed".to_owned(),
                }),
            )
        }
        (
            LinalgExpectedOutcome::Scalar { value, atol, rtol },
            Ok(LinalgObservedOutcome::Scalar { value: got }),
        ) => {
            let md = (*got - *value).abs();
            let pass = allclose_scalar(*got, *value, *atol, *rtol);
            let msg = if pass {
                format!("scalar matched (diff={md:.2e})")
            } else {
                format!("scalar mismatch: diff={md:.2e}, atol={atol}, rtol={rtol}")
            };
            (
                pass,
                msg,
                Some(md),
                Some(ToleranceUsed {
                    atol: *atol,
                    rtol: *rtol,
                    comparison_mode: "mixed".to_owned(),
                }),
            )
        }
        (
            LinalgExpectedOutcome::Lstsq {
                x,
                residuals,
                rank,
                singular_values,
                atol,
                rtol,
            },
            Ok(LinalgObservedOutcome::Lstsq {
                x: got_x,
                residuals: got_res,
                rank: got_rank,
                singular_values: got_s,
            }),
        ) => {
            let md = max_diff_vec(got_x, x)
                .max(max_diff_vec(got_res, residuals))
                .max(max_diff_vec(got_s, singular_values));
            let pass = *rank == *got_rank
                && allclose_vec(got_x, x, *atol, *rtol)
                && allclose_vec(got_res, residuals, *atol, *rtol)
                && allclose_vec(got_s, singular_values, *atol, *rtol);
            let msg = if pass {
                format!("lstsq matched (max_diff={md:.2e})")
            } else {
                format!("lstsq mismatch: max_diff={md:.2e}, rank expected={rank} got={got_rank}")
            };
            (
                pass,
                msg,
                Some(md),
                Some(ToleranceUsed {
                    atol: *atol,
                    rtol: *rtol,
                    comparison_mode: "mixed".to_owned(),
                }),
            )
        }
        (
            LinalgExpectedOutcome::Pinv {
                values,
                rank,
                atol,
                rtol,
            },
            Ok(LinalgObservedOutcome::Pinv {
                values: got,
                rank: got_rank,
            }),
        ) => {
            let md = max_diff_matrix(got, values);
            let pass = *rank == *got_rank && allclose_matrix(got, values, *atol, *rtol);
            let msg = if pass {
                format!("pinv matched (max_diff={md:.2e})")
            } else {
                format!("pinv mismatch: max_diff={md:.2e}, rank expected={rank} got={got_rank}")
            };
            (
                pass,
                msg,
                Some(md),
                Some(ToleranceUsed {
                    atol: *atol,
                    rtol: *rtol,
                    comparison_mode: "mixed".to_owned(),
                }),
            )
        }
        (LinalgExpectedOutcome::Error { error }, Err(actual)) => {
            let pass = matches_error_contract(&actual.to_string(), error);
            let msg = if pass {
                "error matched".to_owned()
            } else {
                format!("error mismatch: expected=`{error}`, got=`{actual}`")
            };
            (pass, msg, None, None)
        }
        (expected, got) => (
            false,
            format!("shape mismatch: expected {expected:?} but observed {got:?}"),
            None,
            None,
        ),
    }
}

/// Compute the maximum absolute difference between two vectors.
fn max_diff_vec(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        })
}

/// Compute the maximum absolute difference between two matrices.
fn max_diff_matrix(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ar, br)| max_diff_vec(ar, br))
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        })
}

/// Probe whether the oracle is available without running a full test.
fn probe_oracle_availability(config: &DifferentialOracleConfig) -> OracleStatus {
    if !config.script_path.exists() {
        return OracleStatus::Missing {
            reason: format!("script not found: {}", config.script_path.display()),
        };
    }

    let result = Command::new(&config.python_path)
        .arg("-c")
        .arg("import scipy; print(scipy.__version__)")
        .output();

    match result {
        Ok(output) if output.status.success() => OracleStatus::Available,
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            if stderr.contains("No module named") {
                OracleStatus::Missing {
                    reason: format!("scipy not available: {stderr}"),
                }
            } else {
                OracleStatus::Failed { reason: stderr }
            }
        }
        Err(e) => OracleStatus::Missing {
            reason: format!("python not found: {e}"),
        },
    }
}

// ═══════════════════════════════════════════════════════════════════
// Packet Runner Registry (§bd-3jh.19.10)
// ═══════════════════════════════════════════════════════════════════

/// Known packet families across the FrankenSciPy conformance surface.
///
/// Each variant maps to one P2C ticket and its associated fixture format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PacketFamily {
    /// P2C-001: Integration tolerance validation
    ValidateTol,
    /// P2C-002: Dense linear algebra (solve, inv, det, lstsq, pinv)
    LinalgCore,
    /// P2C-003: Optimization and root-finding routines
    Optimize,
    /// P2C-004: Sparse matrix operations
    SparseOps,
    /// P2C-005: FFT routines
    Fft,
    /// P2C-006: Special functions
    Special,
    /// P2C-007: Array API compatibility
    ArrayApi,
    /// P2C-008: CASP runtime (PolicyController, SolverPortfolio, ConformalCalibrator)
    RuntimeCasp,
    /// P2C-009: Cluster analysis (linkage, vq, metrics)
    Cluster,
    /// P2C-010: Spatial algorithms and distance metrics
    Spatial,
    /// P2C-011: Signal processing (filters, windows, transforms)
    Signal,
    /// P2C-012: Statistics (descriptive, hypothesis tests, correlations)
    Stats,
    /// P2C-013: Integration (quadrature, ODE solvers)
    IntegrateCore,
    /// P2C-014: Interpolation routines
    InterpolateCore,
    /// P2C-015: ndimage filtering, morphology, measurements, and transforms
    NdimageCore,
    /// P2C-016: physical constants and unit conversions
    ConstantsCore,
    /// P2C-017: Input/output formats and helper routines
    IoCore,
}

impl PacketFamily {
    /// All known packet families for enumeration.
    pub const ALL: [Self; 17] = [
        Self::ValidateTol,
        Self::LinalgCore,
        Self::Optimize,
        Self::SparseOps,
        Self::Fft,
        Self::Special,
        Self::ArrayApi,
        Self::RuntimeCasp,
        Self::Cluster,
        Self::Spatial,
        Self::Signal,
        Self::Stats,
        Self::IntegrateCore,
        Self::InterpolateCore,
        Self::NdimageCore,
        Self::ConstantsCore,
        Self::IoCore,
    ];

    /// Canonical packet ID for this family (e.g., "FSCI-P2C-001").
    #[must_use]
    pub const fn packet_id(&self) -> &'static str {
        match self {
            Self::ValidateTol => "FSCI-P2C-001",
            Self::LinalgCore => "FSCI-P2C-002",
            Self::Optimize => "FSCI-P2C-003",
            Self::SparseOps => "FSCI-P2C-004",
            Self::Fft => "FSCI-P2C-005",
            Self::Special => "FSCI-P2C-006",
            Self::ArrayApi => "FSCI-P2C-007",
            Self::RuntimeCasp => "FSCI-P2C-008",
            Self::Cluster => "FSCI-P2C-009",
            Self::Spatial => "FSCI-P2C-010",
            Self::Signal => "FSCI-P2C-011",
            Self::Stats => "FSCI-P2C-012",
            Self::IntegrateCore => "FSCI-P2C-013",
            Self::InterpolateCore => "FSCI-P2C-014",
            Self::NdimageCore => "FSCI-P2C-015",
            Self::ConstantsCore => "FSCI-P2C-016",
            Self::IoCore => "FSCI-P2C-017",
        }
    }

    /// Short name used in fixture file naming (e.g., "validate_tol").
    #[must_use]
    pub const fn family_name(&self) -> &'static str {
        match self {
            Self::ValidateTol => "validate_tol",
            Self::LinalgCore => "linalg_core",
            Self::Optimize => "optimize",
            Self::SparseOps => "sparse_ops",
            Self::Fft => "fft",
            Self::Special => "special",
            Self::ArrayApi => "array_api",
            Self::RuntimeCasp => "runtime_casp",
            Self::Cluster => "cluster_core",
            Self::Spatial => "spatial_core",
            Self::Signal => "signal_core",
            Self::Stats => "stats_core",
            Self::IntegrateCore => "integrate_core",
            Self::InterpolateCore => "interpolate_core",
            Self::NdimageCore => "ndimage_core",
            Self::ConstantsCore => "constants_core",
            Self::IoCore => "io_core",
        }
    }

    /// Detect family from a fixture family string.
    #[must_use]
    pub fn from_family_str(s: &str) -> Option<Self> {
        if s.contains("validate_tol") {
            Some(Self::ValidateTol)
        } else if s.contains("linalg") {
            Some(Self::LinalgCore)
        } else if s.contains("sparse") {
            Some(Self::SparseOps)
        } else if s.contains("fft") {
            Some(Self::Fft)
        } else if s.contains("special") {
            Some(Self::Special)
        } else if s.contains("optim") {
            Some(Self::Optimize)
        } else if s.contains("array_api") {
            Some(Self::ArrayApi)
        } else if s.contains("runtime") || s.contains("casp") {
            Some(Self::RuntimeCasp)
        } else if s.contains("cluster") {
            Some(Self::Cluster)
        } else if s.contains("spatial") {
            Some(Self::Spatial)
        } else if s.contains("signal") {
            Some(Self::Signal)
        } else if s.contains("stats") {
            Some(Self::Stats)
        } else if s.contains("integrate") {
            Some(Self::IntegrateCore)
        } else if s.contains("interpolate") {
            Some(Self::InterpolateCore)
        } else if s.contains("ndimage") {
            Some(Self::NdimageCore)
        } else if s.contains("constants") {
            Some(Self::ConstantsCore)
        } else if s.contains("io") {
            Some(Self::IoCore)
        } else {
            None
        }
    }

    /// Whether this family has a working runner implementation.
    #[must_use]
    pub const fn has_runner(&self) -> bool {
        matches!(
            self,
            Self::ValidateTol
                | Self::LinalgCore
                | Self::Optimize
                | Self::SparseOps
                | Self::Fft
                | Self::Special
                | Self::ArrayApi
                | Self::RuntimeCasp
                | Self::Cluster
                | Self::Spatial
                | Self::Signal
                | Self::Stats
                | Self::IntegrateCore
                | Self::InterpolateCore
                | Self::NdimageCore
                | Self::ConstantsCore
                | Self::IoCore
        )
    }

    /// Canonical fixture filename for this family.
    #[must_use]
    pub fn fixture_filename(&self) -> String {
        format!("{}_{}.json", self.packet_id(), self.family_name())
    }
}

/// Aggregate parity report merging results from multiple packet families.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateParityReport {
    /// Timestamp of aggregation.
    pub generated_unix_ms: u128,
    /// Per-packet summaries.
    pub packets: Vec<PacketSummary>,
    /// Total cases across all packets.
    pub total_cases: usize,
    /// Total passed across all packets.
    pub total_passed: usize,
    /// Total failed across all packets.
    pub total_failed: usize,
}

/// Aggregate individual packet reports into a cross-packet parity report.
#[must_use]
pub fn aggregate_packet_reports(reports: &[PacketReport]) -> AggregateParityReport {
    let packets: Vec<PacketSummary> = reports.iter().map(packet_summary).collect();
    let total_cases: usize = packets.iter().map(|p| p.total_cases).sum();
    let total_passed: usize = packets.iter().map(|p| p.passed_cases).sum();
    let total_failed: usize = packets.iter().map(|p| p.failed_cases).sum();

    AggregateParityReport {
        generated_unix_ms: now_unix_ms(),
        packets,
        total_cases,
        total_passed,
        total_failed,
    }
}

/// Ensure the canonical artifact directory layout exists for a packet.
///
/// Creates: `{fixture_root}/artifacts/{packet_id}/{subdir}` for each
/// required subdirectory (anchor, contracts, threats).
pub fn ensure_artifact_layout(
    config: &HarnessConfig,
    packet_id: &str,
) -> Result<PathBuf, HarnessError> {
    let base = config.artifact_dir_for(packet_id);
    for subdir in &["anchor", "contracts", "threats"] {
        let dir = base.join(subdir);
        if !dir.exists() {
            fs::create_dir_all(&dir)
                .map_err(|source| HarnessError::ArtifactIo { path: dir, source })?;
        }
    }
    Ok(base)
}

/// Discover all fixture files in the fixture root and return their paths
/// grouped by detected packet family.
pub fn discover_fixtures(
    config: &HarnessConfig,
) -> Result<Vec<(PacketFamily, PathBuf)>, HarnessError> {
    let mut results = Vec::new();
    let entries = fs::read_dir(&config.fixture_root).map_err(|source| HarnessError::FixtureIo {
        path: config.fixture_root.clone(),
        source,
    })?;

    for entry in entries {
        let entry = entry.map_err(|source| HarnessError::FixtureIo {
            path: config.fixture_root.clone(),
            source,
        })?;
        let path = entry.path();
        if path.extension().and_then(OsStr::to_str) != Some("json") {
            continue;
        }
        let name = path.file_stem().and_then(OsStr::to_str).unwrap_or_default();
        // Skip non-fixture files (smoke_case, etc.)
        if !name.starts_with("FSCI-P2C-") {
            continue;
        }
        // Try to detect family from filename
        if let Some(family) = PacketFamily::from_family_str(name) {
            results.push((family, path));
        }
    }

    results.sort_by(|a, b| a.0.packet_id().cmp(b.0.packet_id()));
    Ok(results)
}

/// Run all discoverable fixtures and produce an aggregate report.
pub fn run_all_packets(config: &HarnessConfig) -> Result<AggregateParityReport, HarnessError> {
    let fixtures = discover_fixtures(config)?;
    let mut reports = Vec::new();

    for (family, path) in &fixtures {
        if !family.has_runner() {
            continue;
        }
        let fixture_name = path.file_name().and_then(OsStr::to_str).unwrap_or_default();
        let report = match family {
            PacketFamily::ValidateTol => run_validate_tol_packet(config, fixture_name)?,
            PacketFamily::LinalgCore => run_linalg_packet(config, fixture_name)?,
            PacketFamily::Optimize => run_optimize_packet(config, fixture_name)?,
            PacketFamily::SparseOps => run_sparse_packet(config, fixture_name)?,
            PacketFamily::Fft => run_fft_packet(config, fixture_name)?,
            PacketFamily::Special => run_special_packet(config, fixture_name)?,
            PacketFamily::ArrayApi => run_array_api_packet(config, fixture_name)?,
            PacketFamily::RuntimeCasp => run_casp_packet(config, fixture_name)?,
            PacketFamily::Cluster => run_cluster_packet(config, fixture_name)?,
            PacketFamily::Spatial => run_spatial_packet(config, fixture_name)?,
            PacketFamily::Signal => run_signal_packet(config, fixture_name)?,
            PacketFamily::Stats => run_stats_packet(config, fixture_name)?,
            PacketFamily::IntegrateCore => run_integrate_packet(config, fixture_name)?,
            PacketFamily::InterpolateCore => run_interpolate_packet(config, fixture_name)?,
            PacketFamily::NdimageCore => run_ndimage_packet(config, fixture_name)?,
            PacketFamily::ConstantsCore => run_constants_packet(config, fixture_name)?,
            PacketFamily::IoCore => run_io_packet(config, fixture_name)?,
        };
        reports.push(report);
    }

    Ok(aggregate_packet_reports(&reports))
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "dashboard")]
    use super::style_for_case_result;
    use super::{
        AggregateParityReport, ArrayApiExpectedOutcome, ArrayApiPacketFixture, CaspPacketFixture,
        ClusterPacketFixture, ConformanceReport, DifferentialCaseResult, DifferentialOracleConfig,
        DriftDiffReport, DriftGateThreshold, FftPacketFixture, HarnessConfig,
        IntegratePacketFixture, IoPacketFixture, LinalgCase, LinalgExpectedOutcome,
        LinalgPacketFixture, LiveOracleCaptureReport, OptimizePacketFixture, OracleCaseOutput,
        OracleStatus, PacketFamily, PacketReport, PythonOracleConfig, SignalPacketFixture,
        SparsePacketFixture, SpatialPacketFixture, SpecialCase, SpecialCaseFunction,
        SpecialExpectedOutcome, SpecialPacketFixture, StatsCase, StatsExpected, StatsObserved,
        StatsPacketFixture, ToleranceUsed, aggregate_packet_reports, build_drift_diff_report,
        compare_linalg_case_against_oracle, compare_stats_case_against_oracle,
        default_zero_drift_thresholds, discover_differential_fixture_paths, discover_fixtures,
        ensure_artifact_layout, evaluate_drift_gate, load_array_api_contract_table,
        load_oracle_capture, resolve_array_api_contract_tolerance, run_array_api_packet,
        run_casp_packet, run_cluster_packet, run_differential_test, run_fft_packet,
        run_integrate_packet, run_interpolate_packet, run_io_packet, run_linalg_packet,
        run_linalg_packet_with_oracle_capture, run_live_oracle_capture_lane, run_ndimage_packet,
        run_optimize_packet, run_signal_packet, run_smoke, run_sparse_packet, run_spatial_packet,
        run_special_packet, run_stats_packet, run_validate_tol_packet,
        validate_zero_drift_oracle_coverage, write_differential_parity_artifacts,
        write_drift_diff_artifact, write_parity_artifacts,
    };
    use fsci_linalg::LinalgError;
    use fsci_runtime::RuntimeMode;
    use serde::Serialize;
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;

    const INTERPOLATE_CORE_CASE_COUNT: usize = 24;

    #[test]
    fn smoke_harness_finds_oracle_and_fixtures() {
        let cfg = HarnessConfig::default_paths();
        let report = run_smoke(&cfg).expect("smoke packet should run");
        assert!(report.oracle_present, "oracle repo should be present");
        assert!(report.cases_run >= 1, "expected at least one executed case");
        assert_eq!(report.failed_cases, 0, "smoke packet should pass");
        assert!(report.strict_mode);
    }

    #[test]
    fn matches_error_contract_normalizes_case_and_whitespace() {
        // Exact match still passes.
        assert!(super::matches_error_contract(
            "tolerance must be non-negative",
            "tolerance must be non-negative"
        ));
        // Case-insensitive.
        assert!(super::matches_error_contract(
            "Tolerance must be non-negative",
            "tolerance must be non-negative"
        ));
        // Whitespace-normalized.
        assert!(super::matches_error_contract(
            "tolerance   must\tbe  non-negative",
            "tolerance must be non-negative"
        ));
        // Expected as substring of actual (e.g., actual has a prefix like
        // "IoError(InvalidArgument {...}): ...").
        assert!(super::matches_error_contract(
            "LinalgError::SingularMatrix: matrix is singular",
            "matrix is singular"
        ));
        // Empty expected passes (used for 'any error').
        assert!(super::matches_error_contract("whatever", ""));
        // Non-overlapping text fails.
        assert!(!super::matches_error_contract(
            "tolerance must be positive",
            "domain error"
        ));
    }

    #[test]
    fn verify_oracle_capture_provenance_rejects_stale_fixture() {
        use super::{OracleCapture, OracleCaptureProvenance, verify_oracle_capture_provenance};
        let fixture_raw = b"{\"packet_id\":\"P2C-001\",\"cases\":[]}";
        let current_hash = blake3::hash(fixture_raw).to_hex().to_string();

        // A capture recorded with the CURRENT fixture hash: verification passes.
        let fresh_capture = OracleCapture {
            packet_id: "P2C-001".into(),
            family: "test".into(),
            generated_unix_ms: 0,
            runtime: None,
            case_outputs: Vec::new(),
            provenance: Some(OracleCaptureProvenance {
                fixture_input_blake3: current_hash.clone(),
                oracle_output_blake3: "0".into(),
                capture_blake3: "0".into(),
            }),
        };
        assert!(
            super::verify_oracle_capture_provenance(
                &fresh_capture,
                fixture_raw,
                "test_fixture.json"
            )
            .is_ok()
        );

        // A capture whose stored hash doesn't match: rejected.
        let stale_capture = OracleCapture {
            packet_id: "P2C-001".into(),
            family: "test".into(),
            generated_unix_ms: 0,
            runtime: None,
            case_outputs: Vec::new(),
            provenance: Some(OracleCaptureProvenance {
                fixture_input_blake3: "deadbeef".into(),
                oracle_output_blake3: "0".into(),
                capture_blake3: "0".into(),
            }),
        };
        let err =
            verify_oracle_capture_provenance(&stale_capture, fixture_raw, "test_fixture.json")
                .expect_err("stale fixture must fail");
        assert!(err.to_string().contains("stale oracle capture"));

        // Legacy capture without provenance: no-op passes.
        let legacy_capture = OracleCapture {
            packet_id: "P2C-001".into(),
            family: "test".into(),
            generated_unix_ms: 0,
            runtime: None,
            case_outputs: Vec::new(),
            provenance: None,
        };
        assert!(
            super::verify_oracle_capture_provenance(
                &legacy_capture,
                fixture_raw,
                "test_fixture.json"
            )
            .is_ok()
        );
    }

    #[test]
    fn allclose_scalar_preserves_infinity_sign() {
        assert!(super::allclose_scalar(
            f64::INFINITY,
            f64::INFINITY,
            1.0e-12,
            1.0e-12
        ));
        assert!(super::allclose_scalar(
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            1.0e-12,
            1.0e-12
        ));
        assert!(!super::allclose_scalar(
            f64::INFINITY,
            f64::NEG_INFINITY,
            1.0e-12,
            1.0e-12
        ));
        assert!(!super::allclose_scalar(
            f64::INFINITY,
            1.0e308,
            1.0e-12,
            1.0e-12
        ));
        assert!(!super::allclose_scalar(
            1.0e308,
            f64::NEG_INFINITY,
            1.0e-12,
            1.0e-12
        ));
    }

    #[test]
    fn allclose_scalar_keeps_equal_nan_behavior() {
        assert!(super::allclose_scalar(f64::NAN, f64::NAN, 1.0e-12, 1.0e-12));
        assert!(!super::allclose_scalar(f64::NAN, 0.0, 1.0e-12, 1.0e-12));
        assert!(!super::allclose_scalar(0.0, f64::NAN, 1.0e-12, 1.0e-12));
    }

    #[test]
    fn special_fixture_schema_errors_do_not_match_domain_errors() {
        let case = super::SpecialCase {
            case_id: "fixture_arity".to_owned(),
            category: "differential".to_owned(),
            mode: RuntimeMode::Strict,
            function: super::SpecialCaseFunction::Gamma,
            args: vec![],
            expected: super::SpecialExpectedOutcome::ErrorKind {
                error: "DomainError".to_owned(),
            },
        };
        let observed = Err(super::special_invalid_fixture_error(
            "gamma",
            RuntimeMode::Strict,
        ));

        let (passed, message, _, _) = super::compare_special_case_differential(&case, &observed);

        assert!(!passed);
        assert!(message.contains("expected=DomainError"));
        assert!(message.contains("FixtureSchemaError"));
    }

    #[test]
    fn signal_compare_uses_allclose_scalar_for_nonfinite_values() {
        let case = super::SignalCase {
            case_id: "signal-nonfinite".to_owned(),
            category: "unit".to_owned(),
            mode: RuntimeMode::Strict,
            function: "window".to_owned(),
            args: Vec::new(),
            expected: super::SignalExpected {
                kind: "array".to_owned(),
                value: Some(vec![f64::INFINITY, f64::NAN]),
                b: None,
                a: None,
                w: None,
                h_mag: None,
                h_phase: None,
                real: None,
                imag: None,
                atol: Some(1.0e-12),
                rtol: Some(1.0e-12),
                contract_ref: String::new(),
                error: None,
            },
        };

        let observed = super::SignalObserved::Array(vec![f64::INFINITY, f64::NAN]);
        let (passed, message) = super::compare_signal_outcome(&case, &observed);
        assert!(passed, "{message}");

        let wrong_sign = super::SignalObserved::Array(vec![f64::NEG_INFINITY, f64::NAN]);
        let (passed, message) = super::compare_signal_outcome(&case, &wrong_sign);
        assert!(!passed, "{message}");
    }

    #[test]
    fn points_close_uses_allclose_scalar_for_nonfinite_values() {
        assert!(super::points_close(
            &[f64::INFINITY, f64::NAN],
            &[f64::INFINITY, f64::NAN],
            1.0e-12,
            1.0e-12
        ));
        assert!(!super::points_close(
            &[f64::NEG_INFINITY, f64::NAN],
            &[f64::INFINITY, f64::NAN],
            1.0e-12,
            1.0e-12
        ));
    }

    #[test]
    fn stats_compare_uses_allclose_scalar_for_nonfinite_values() {
        let inf_case = StatsCase {
            case_id: "stats-inf".to_owned(),
            category: "summary".to_owned(),
            mode: RuntimeMode::Strict,
            function: "sem".to_owned(),
            args: vec![serde_json::json!([1.0, 2.0, 3.0])],
            expected: StatsExpected {
                kind: "scalar".to_owned(),
                value: Some(f64::INFINITY),
                nobs: None,
                minmax: None,
                mean: None,
                variance: None,
                skewness: None,
                kurtosis: None,
                statistic: None,
                pvalue: None,
                slope: None,
                intercept: None,
                rvalue: None,
                stderr: None,
                array_value: None,
                atol: Some(1.0e-12),
                rtol: Some(1.0e-12),
                contract_ref: String::new(),
            },
        };
        let observed = StatsObserved::Scalar(f64::INFINITY);
        let (passed, message) = super::compare_stats_outcome(&inf_case, &observed);
        assert!(passed, "{message}");

        let wrong_sign = StatsObserved::Scalar(f64::NEG_INFINITY);
        let (passed, message) = super::compare_stats_outcome(&inf_case, &wrong_sign);
        assert!(!passed, "{message}");

        let nan_case = StatsCase {
            case_id: "stats-nan".to_owned(),
            category: "summary".to_owned(),
            mode: RuntimeMode::Strict,
            function: "sem".to_owned(),
            args: vec![serde_json::json!([1.0, 2.0, 3.0])],
            expected: StatsExpected {
                kind: "scalar".to_owned(),
                value: Some(f64::NAN),
                nobs: None,
                minmax: None,
                mean: None,
                variance: None,
                skewness: None,
                kurtosis: None,
                statistic: None,
                pvalue: None,
                slope: None,
                intercept: None,
                rvalue: None,
                stderr: None,
                array_value: None,
                atol: Some(1.0e-12),
                rtol: Some(1.0e-12),
                contract_ref: String::new(),
            },
        };
        let observed = StatsObserved::Scalar(f64::NAN);
        let (passed, message) = super::compare_stats_outcome(&nan_case, &observed);
        assert!(passed, "{message}");
    }

    #[test]
    fn validate_tol_packet_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_validate_tol_packet(&cfg, "FSCI-P2C-001_validate_tol.json")
            .expect("packet fixture should run");
        assert_eq!(report.failed_cases, 0);
        assert!(report.passed_cases >= 1);

        let artifacts =
            write_parity_artifacts(&cfg, &report).expect("artifact generation must pass");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());

        #[cfg(feature = "dashboard")]
        {
            let styled = style_for_case_result(report.case_results.first().expect("case exists"));
            assert!(
                styled.fg.is_some(),
                "style must be colorized for dashboard rendering"
            );
        }
    }

    #[test]
    fn write_parity_artifacts_records_real_decode_proof() {
        let root = PathBuf::from("/tmp").join(format!(
            "fsci-conformance-decode-proof-{}",
            super::now_unix_ms()
        ));
        let cfg = HarnessConfig {
            oracle_root: root.join("oracle"),
            fixture_root: root.join("fixtures"),
            strict_mode: true,
        };
        let report = PacketReport {
            schema_version: super::packet_report_schema_v2(),
            packet_id: "REAL-DECODE-PROOF".to_owned(),
            family: "synthetic".to_owned(),
            case_results: vec![
                super::CaseResult {
                    case_id: "case_ok".to_owned(),
                    passed: true,
                    message: "synthetic pass".to_owned(),
                },
                super::CaseResult {
                    case_id: "case_fail".to_owned(),
                    passed: false,
                    message: "synthetic fail".to_owned(),
                },
            ],
            passed_cases: 1,
            failed_cases: 1,
            fixture_path: None,
            oracle_status: None,
            differential_case_results: None,
            report_kind: super::ReportKind::Unspecified,
            generated_unix_ms: super::now_unix_ms(),
        };

        let artifacts =
            write_parity_artifacts(&cfg, &report).expect("artifact generation must pass");
        let report_bytes = fs::read(&artifacts.report_path).expect("read parity report");
        let decode_proof: super::DecodeProofArtifact =
            serde_json::from_slice(&fs::read(&artifacts.decode_proof_path).expect("read proof"))
                .expect("decode proof should parse");

        assert_eq!(decode_proof.recovered_blocks, 1);
        assert!(
            decode_proof.reason.contains("simulated corruption"),
            "decode proof should describe the recovery drill"
        );
        assert_eq!(
            decode_proof.proof_hash,
            blake3::hash(&report_bytes).to_hex().to_string()
        );
    }

    #[test]
    fn write_parity_artifacts_is_byte_stable_modulo_generated_unix_ms() {
        // /testing-golden-artifacts regression for [frankenscipy-okuhi]:
        // when the only difference between the existing on-disk
        // parity_report.json and the freshly-built report is the
        // generated_unix_ms field, the writer must reuse the on-disk
        // bytes verbatim. Otherwise every test run would rewrite three
        // committed files (parity_report.json + .raptorq.json sidecar +
        // .decode_proof.json) with no semantic difference.
        let root =
            PathBuf::from("/tmp").join(format!("fsci-conformance-okuhi-{}", super::now_unix_ms()));
        let cfg = HarnessConfig {
            oracle_root: root.join("oracle"),
            fixture_root: root.join("fixtures"),
            strict_mode: true,
        };
        let mk_report = |ts: u128| PacketReport {
            schema_version: super::packet_report_schema_v2(),
            packet_id: "REGRESSION-OKUHI".to_owned(),
            family: "synthetic".to_owned(),
            case_results: vec![super::CaseResult {
                case_id: "stable".to_owned(),
                passed: true,
                message: "deterministic".to_owned(),
            }],
            passed_cases: 1,
            failed_cases: 0,
            fixture_path: None,
            oracle_status: None,
            differential_case_results: None,
            report_kind: super::ReportKind::Unspecified,
            generated_unix_ms: ts,
        };

        // First write — fresh dir, must produce all three artifacts.
        let r1 = mk_report(1_000);
        let a1 = write_parity_artifacts(&cfg, &r1).expect("first write must succeed");
        let bytes_report_1 = fs::read(&a1.report_path).expect("read parity_report.json #1");
        let bytes_sidecar_1 = fs::read(&a1.sidecar_path).expect("read raptorq sidecar #1");
        let bytes_proof_1 = fs::read(&a1.decode_proof_path).expect("read decode proof #1");

        // Second write — same content, different timestamp.
        // All three on-disk files must be byte-identical to the first
        // run because the writer reuses the existing parity_report.json
        // (and the sidecars are hash-derived from those bytes).
        let r2 = mk_report(2_000);
        let a2 = write_parity_artifacts(&cfg, &r2).expect("second write must succeed");
        let bytes_report_2 = fs::read(&a2.report_path).expect("read parity_report.json #2");
        let bytes_sidecar_2 = fs::read(&a2.sidecar_path).expect("read raptorq sidecar #2");
        let bytes_proof_2 = fs::read(&a2.decode_proof_path).expect("read decode proof #2");

        assert_eq!(
            bytes_report_1, bytes_report_2,
            "parity_report.json must be byte-stable when only generated_unix_ms differs"
        );
        assert_eq!(
            bytes_sidecar_1, bytes_sidecar_2,
            "parity_report.raptorq.json must be byte-stable when source is byte-stable"
        );
        assert_eq!(
            bytes_proof_1, bytes_proof_2,
            "parity_report.decode_proof.json must be byte-stable when source is byte-stable"
        );

        // Third write — content actually changed (different message).
        // The writer must produce fresh bytes.
        let mut r3 = mk_report(3_000);
        r3.case_results[0].message = "different content".to_owned();
        write_parity_artifacts(&cfg, &r3).expect("third write must succeed");
        let bytes_report_3 = fs::read(&a1.report_path).expect("read parity_report.json #3");
        assert_ne!(
            bytes_report_1, bytes_report_3,
            "writer must not be sticky when actual content changes"
        );
    }

    #[test]
    fn linalg_packet_passes() {
        let cfg = HarnessConfig::default_paths();
        let report =
            run_linalg_packet(&cfg, "FSCI-P2C-002_linalg_core.json").expect("linalg packet runs");
        if report.failed_cases > 0 {
            for res in &report.case_results {
                if !res.passed {
                    println!("FAILED CASE: {:#?}", res);
                }
            }
        }
        assert_eq!(report.failed_cases, 0);
        assert!(report.passed_cases >= 1);

        let artifacts =
            write_parity_artifacts(&cfg, &report).expect("linalg parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn generate_decode_proof_artifact_recovered_blocks_matches_drop_count() {
        // REVIEW MODE [LOW] regression: recovered_blocks used to be a
        // magic constant separately maintained from drop_indices. After
        // the fix it's derived from drop_indices.len(), so the public
        // API's recovered_blocks always reflects what was actually
        // dropped+recovered. Pin the contract: a single-block drop
        // produces recovered_blocks == 1 and a singular reason string.
        let payload: Vec<u8> = (0..2048u32)
            .flat_map(|i| (i as u16).to_le_bytes())
            .collect();
        let sidecar = super::generate_raptorq_sidecar(&payload).expect("sidecar");
        let proof = super::generate_decode_proof_artifact(&payload, &sidecar)
            .expect("decode proof should succeed on a fresh sidecar");
        assert_eq!(
            proof.recovered_blocks, 1,
            "default decode-proof drops one block"
        );
        assert!(
            proof.reason.contains("1 block"),
            "reason should be singular for 1 block, got {:?}",
            proof.reason
        );
        // proof_hash equals payload hash (the recovery round-trip).
        let expected = super::hash(&payload).to_hex().to_string();
        assert_eq!(
            proof.proof_hash, expected,
            "decode-proof hash must round-trip"
        );
    }

    #[test]
    fn recover_payload_with_sidecar_rejects_zero_symbol_size_sidecar() {
        // REVIEW MODE [LOW] regression: RaptorQSidecar is serde-derived
        // and may be loaded from external JSON. A tampered file with
        // symbol_size=0 used to crash chunk_payload (div by zero). The
        // fix returns a typed HarnessError before any panicky path.
        let payload = b"any payload, doesn't matter";
        let bogus_sidecar = super::RaptorQSidecar {
            schema_version: 1,
            source_hash: "0".repeat(64),
            symbol_size: 0,
            seed: 0,
            source_symbols: 1,
            repair_symbols: 0,
            repair_symbol_hashes: Vec::new(),
            repair_symbol_payloads_hex: Vec::new(),
        };
        let err = super::recover_payload_with_sidecar(payload, &bogus_sidecar, &[])
            .expect_err("symbol_size=0 must be rejected, not panic");
        let msg = err.to_string();
        assert!(
            msg.contains("symbol_size"),
            "error message should name the failed field; got {msg:?}"
        );
    }

    #[test]
    fn recover_payload_with_sidecar_rejects_out_of_range_drop_index() {
        // REVIEW MODE [LOW] regression: recover_payload_with_sidecar
        // used to silently skip out-of-range entries in drop_indices,
        // which would let a caller pass a stale/wrong index and still
        // get a "decode proof" without exercising recovery. The fix
        // returns an explicit RaptorQ error for any idx >= k.
        let payload = b"hello, raptorq decode proof regression test"
            .iter()
            .copied()
            .cycle()
            .take(1024)
            .collect::<Vec<u8>>();
        let sidecar = super::generate_raptorq_sidecar(&payload).expect("generate sidecar");
        let k = sidecar.source_symbols;
        // k+5 is unambiguously out of range — must fail loud.
        let bogus_idx = k + 5;
        let err = super::recover_payload_with_sidecar(&payload, &sidecar, &[bogus_idx])
            .expect_err("out-of-range drop_indices must error");
        let msg = err.to_string();
        assert!(
            msg.contains("out-of-range"),
            "error message should name the failure class; got {msg:?}"
        );
        assert!(
            msg.contains(&bogus_idx.to_string()),
            "error message should include the bogus index; got {msg:?}"
        );
    }

    #[test]
    fn linalg_fixture_accepts_nonfinite_string_sentinels() {
        let raw = r#"{
            "packet_id": "FSCI-P2C-002",
            "family": "linalg_core",
            "cases": [
                {
                    "operation": "solve",
                    "case_id": "solve_nonfinite_sentinel",
                    "mode": "Hardened",
                    "a": [["NaN", 0.0], [0.0, "-Infinity"]],
                    "b": ["Infinity", 1.0],
                    "expected": {
                        "kind": "error",
                        "error": "array must not contain infs or NaNs"
                    }
                }
            ]
        }"#;

        let fixture: LinalgPacketFixture =
            serde_json::from_str(raw).expect("linalg sentinel fixture should parse");
        assert!(
            matches!(&fixture.cases[0], LinalgCase::Solve { .. }),
            "expected solve case"
        );
        if let LinalgCase::Solve { a, b, .. } = &fixture.cases[0] {
            assert!(a[0][0].is_nan());
            assert_eq!(a[1][1], f64::NEG_INFINITY);
            assert_eq!(b[0], f64::INFINITY);
        }

        let encoded = serde_json::to_string(&fixture).expect("fixture should serialize");
        assert!(encoded.contains("\"NaN\""));
        assert!(encoded.contains("\"Infinity\""));
        assert!(encoded.contains("\"-Infinity\""));
    }

    #[test]
    fn sparse_fixture_accepts_nonfinite_string_sentinels() {
        let raw = r#"{
            "packet_id": "FSCI-P2C-004",
            "family": "sparse_ops",
            "cases": [
                {
                    "case_id": "sparse_nonfinite_sentinel",
                    "category": "spsolve",
                    "mode": "Hardened",
                    "operation": "spsolve",
                    "format": "csr",
                    "rows": 2,
                    "cols": 2,
                    "data": ["NaN", 1.0],
                    "row_indices": [0, 1],
                    "col_indices": [0, 1],
                    "rhs": ["Infinity", 2.0],
                    "scalar": "-Infinity",
                    "expected": {
                        "kind": "error",
                        "error": "matrix/rhs contains NaN or Inf"
                    }
                }
            ]
        }"#;

        let fixture: SparsePacketFixture =
            serde_json::from_str(raw).expect("sparse sentinel fixture should parse");
        let case = &fixture.cases[0];
        assert!(case.data[0].is_nan());
        assert_eq!(case.rhs.as_ref().expect("rhs")[0], f64::INFINITY);
        assert_eq!(case.scalar, Some(f64::NEG_INFINITY));

        let encoded = serde_json::to_string(&fixture).expect("fixture should serialize");
        assert!(encoded.contains("\"NaN\""));
        assert!(encoded.contains("\"Infinity\""));
        assert!(encoded.contains("\"-Infinity\""));
    }

    #[test]
    fn mock_python_oracle_capture_parses() {
        let unique = format!("fsci-conformance-test-rel-gamma-{}", super::now_unix_ms());
        let root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&root).expect("create temp root");
        let fixtures = root.join("fixtures");
        fs::create_dir_all(&fixtures).expect("create fixtures");

        let fixture_src = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-002_linalg_core.json");
        let fixture_dst = fixtures.join("FSCI-P2C-002_linalg_core.json");
        fs::copy(&fixture_src, &fixture_dst).expect("copy fixture");

        let script_path = root.join("mock_oracle.py");
        let script = r#"
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--fixture", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--oracle-root", required=True)
args = parser.parse_args()

fixture = json.loads(Path(args.fixture).read_text())
result = {
    "packet_id": fixture["packet_id"],
    "family": fixture["family"],
    "generated_unix_ms": 0,
    "runtime": {
        "python_version": "3.11.0",
        "numpy_version": "2.0.0",
        "scipy_version": "mock-1.0",
    },
    "case_outputs": [
        {
            "case_id": c["case_id"],
            "status": "ok",
            "result_kind": "mock",
            "result": {"oracle": "mock"},
            "error": None,
        }
        for c in fixture["cases"]
    ],
}
Path(args.output).write_text(json.dumps(result, indent=2))
"#;
        fs::write(&script_path, script).expect("write script");

        let cfg = HarnessConfig {
            oracle_root: PathBuf::from("/tmp/nonexistent-oracle"),
            fixture_root: fixtures,
            strict_mode: true,
        };
        let oracle = PythonOracleConfig {
            python_bin: PathBuf::from("python3"),
            script_path,
            required: true,
        };

        let output_path =
            super::capture_linalg_oracle(&cfg, "FSCI-P2C-002_linalg_core.json", &oracle)
                .expect("mock oracle capture succeeds");
        let parsed = load_oracle_capture(&output_path).expect("oracle capture parse succeeds");
        assert_eq!(parsed.packet_id, "FSCI-P2C-002");
        assert!(!parsed.case_outputs.is_empty());
        assert_eq!(
            parsed
                .runtime
                .as_ref()
                .expect("runtime metadata should be present")
                .scipy_version,
            "mock-1.0"
        );
        let provenance = parsed
            .provenance
            .as_ref()
            .expect("provenance must be attached after capture");
        assert!(!provenance.fixture_input_blake3.is_empty());
        assert!(!provenance.oracle_output_blake3.is_empty());
        assert!(!provenance.capture_blake3.is_empty());

        let fixture_raw = fs::read_to_string(fixture_dst).expect("read fixture");
        let fixture: LinalgPacketFixture =
            serde_json::from_str(&fixture_raw).expect("fixture parse");
        assert_eq!(parsed.case_outputs.len(), fixture.cases.len());
    }

    #[test]
    fn scipy_optimize_oracle_declared_derivatives_match_finite_differences() {
        let script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("python_oracle/scipy_optimize_oracle.py");

        let output = Command::new("python3")
            .arg(&script_path)
            .arg("--self-check")
            .output()
            .expect("run scipy optimize oracle self-check");
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("No module named") {
                eprintln!("skipping optimize oracle self-check: {stderr}");
                return;
            }
            assert!(
                output.status.success(),
                "optimize oracle self-check should pass\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                stderr
            );
        }

        assert!(
            String::from_utf8_lossy(&output.stdout)
                .contains("optimize oracle derivative self-check passed"),
            "unexpected optimize oracle self-check output:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn scipy_fft_oracle_preserves_complex_paths_and_split_shapes() {
        let script_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python_oracle/scipy_fft_oracle.py");

        let output = Command::new("python3")
            .arg(&script_path)
            .arg("--self-check")
            .output()
            .expect("run scipy fft oracle self-check");
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("No module named") {
                eprintln!("skipping fft oracle self-check: {stderr}");
                return;
            }
            assert!(
                output.status.success(),
                "fft oracle self-check should pass\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                stderr
            );
        }

        assert!(
            String::from_utf8_lossy(&output.stdout)
                .contains("fft oracle shape/complex self-check passed"),
            "unexpected fft oracle self-check output:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn scipy_optimize_oracle_rejects_unknown_objective_and_root_names() {
        let unique = format!("fsci-conformance-test-opt-{}", super::now_unix_ms());
        let root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&root).expect("create temp root");
        let pythonpath = root.join("pythonpath");
        let fake_scipy = pythonpath.join("scipy");
        fs::create_dir_all(&fake_scipy).expect("create fake scipy package");

        fs::write(
            pythonpath.join("numpy.py"),
            r#"__version__ = "mock-numpy-1.0"

def array(values, dtype=None):
    return list(values)

def __getattr__(name):
    def _placeholder(*_args, **_kwargs):
        raise RuntimeError(f"unexpected numpy access: {name}")
    return _placeholder
"#,
        )
        .expect("write fake numpy module");
        fs::write(
            fake_scipy.join("__init__.py"),
            r#"from . import optimize

__version__ = "mock-scipy-1.0"
"#,
        )
        .expect("write fake scipy package");
        fs::write(
            fake_scipy.join("optimize.py"),
            r#"def minimize(*_args, **_kwargs):
    raise RuntimeError("minimize should not run for unknown objective")

def root_scalar(*_args, **_kwargs):
    raise RuntimeError("root_scalar should not run for unknown root function")
"#,
        )
        .expect("write fake scipy.optimize module");

        let fixture_path = root.join("fixture.json");
        let output_path = root.join("oracle.json");
        let fixture = serde_json::json!({
            "packet_id": "FSCI-P2C-003",
            "family": "optimize",
            "cases": [
                {
                    "case_id": "unknown_objective",
                    "operation": "minimize",
                    "objective": "totally_unknown",
                    "method": "Powell",
                    "x0": [0.0, 0.0]
                },
                {
                    "case_id": "unknown_root",
                    "operation": "root",
                    "objective": "totally_unknown_root",
                    "method": "Brentq",
                    "bracket": [0.0, 1.0]
                }
            ]
        });
        fs::write(
            &fixture_path,
            serde_json::to_string_pretty(&fixture).expect("serialize fixture"),
        )
        .expect("write fixture");

        let script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("python_oracle/scipy_optimize_oracle.py");
        let output = Command::new("python3")
            .arg(&script_path)
            .arg("--fixture")
            .arg(&fixture_path)
            .arg("--output")
            .arg(&output_path)
            .arg("--oracle-root")
            .arg(&root)
            .env("PYTHONPATH", &pythonpath)
            .output()
            .expect("run scipy optimize oracle script");

        assert!(
            output.status.success(),
            "optimize oracle script should complete even when cases fail\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let raw = fs::read_to_string(&output_path).expect("read oracle output");
        let payload: serde_json::Value =
            serde_json::from_str(&raw).expect("parse oracle output json");
        let cases = payload["case_outputs"]
            .as_array()
            .expect("case_outputs should be an array");

        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0]["status"], "error");
        assert!(
            cases[0]["error"]
                .as_str()
                .expect("error should be a string")
                .contains("unknown objective"),
            "unexpected objective error payload: {:?}",
            cases[0]
        );
        assert_eq!(cases[1]["status"], "error");
        assert!(
            cases[1]["error"]
                .as_str()
                .expect("error should be a string")
                .contains("unknown root function"),
            "unexpected root error payload: {:?}",
            cases[1]
        );
    }

    #[test]
    fn scipy_special_oracle_rejects_stirling2_until_verified_dispatch_exists() {
        let unique = format!("fsci-conformance-test-rel-jn-{}", super::now_unix_ms());
        let root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&root).expect("create temp root");
        let pythonpath = root.join("pythonpath");
        let fake_scipy = pythonpath.join("scipy");
        fs::create_dir_all(&fake_scipy).expect("create fake scipy package");

        fs::write(
            pythonpath.join("numpy.py"),
            r#"__version__ = "mock-numpy-1.0"

def isscalar(_value):
    return True

def __getattr__(name):
    def _placeholder(*_args, **_kwargs):
        raise RuntimeError(f"unexpected numpy access: {name}")
    return _placeholder
"#,
        )
        .expect("write fake numpy module");
        fs::write(
            fake_scipy.join("__init__.py"),
            r#"from . import special

__version__ = "mock-scipy-1.0"
"#,
        )
        .expect("write fake scipy package");
        fs::write(
            fake_scipy.join("special.py"),
            r#"def __getattr__(name):
    def _placeholder(*_args, **_kwargs):
        raise RuntimeError(f"unexpected scipy.special access: {name}")
    return _placeholder
"#,
        )
        .expect("write fake scipy.special module");

        let fixture_path = root.join("fixture.json");
        let output_path = root.join("oracle.json");
        let fixture = serde_json::json!({
            "packet_id": "TEST-STIRLING2",
            "family": "special_core",
            "cases": [
                {
                    "case_id": "stirling2-case",
                    "function": "stirling2",
                    "args": [3, 2]
                }
            ]
        });
        fs::write(
            &fixture_path,
            serde_json::to_vec_pretty(&fixture).expect("serialize stirling2 fixture"),
        )
        .expect("write fixture");

        let script_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python_oracle/scipy_special_oracle.py");
        let status = Command::new("python3")
            .env("PYTHONPATH", &pythonpath)
            .arg(&script_path)
            .arg("--fixture")
            .arg(&fixture_path)
            .arg("--output")
            .arg(&output_path)
            .arg("--oracle-root")
            .arg("/tmp/nonexistent-oracle")
            .status()
            .expect("run scipy special oracle");
        assert!(status.success(), "oracle script should exit successfully");

        let payload: serde_json::Value =
            serde_json::from_slice(&fs::read(&output_path).expect("read oracle output"))
                .expect("parse oracle output");
        let output = &payload["case_outputs"][0];
        assert_eq!(output["case_id"], "stirling2-case");
        assert_eq!(output["status"], "error");
        assert_eq!(output["result_kind"], "unsupported_function");
        assert!(
            output["error"]
                .as_str()
                .expect("string error")
                .contains("stirling2"),
            "unexpected oracle error payload: {output:?}"
        );
    }

    #[test]
    fn scipy_special_oracle_dispatches_extended_special_func_map_entries() {
        let unique = format!("fsci-conformance-test-special-map-{}", super::now_unix_ms());
        let root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&root).expect("create temp root");
        let pythonpath = root.join("pythonpath");
        let fake_scipy = pythonpath.join("scipy");
        fs::create_dir_all(&fake_scipy).expect("create fake scipy package");

        fs::write(
            pythonpath.join("numpy.py"),
            r#"__version__ = "mock-numpy-1.0"

class _ErrState:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False

def asarray(value, dtype=None):
    if dtype is float:
        return float(value)
    return value

nan = float("nan")

def isscalar(value):
    return not isinstance(value, (list, tuple, dict))

def iscomplexobj(value):
    if isinstance(value, (list, tuple)):
        return any(iscomplexobj(item) for item in value)
    return isinstance(value, complex)

def where(condition, when_true, when_false):
    return when_true if condition else when_false

def log1p(value):
    return value

def expm1(value):
    return value

def sinc(value):
    return value

def errstate(**_kwargs):
    return _ErrState()

def __getattr__(name):
    def _placeholder(*args, **_kwargs):
        if args:
            return args[0]
        raise RuntimeError(f"unexpected numpy access: {name}")
    return _placeholder
"#,
        )
        .expect("write fake numpy module");
        fs::write(
            fake_scipy.join("__init__.py"),
            r#"from . import special

__version__ = "mock-scipy-1.0"
"#,
        )
        .expect("write fake scipy package");
        fs::write(
            fake_scipy.join("special.py"),
            r#"def hyp2f1(a, b, c, z):
    return 3.5

def wright_bessel(a, b, x):
    return 4.5

def spherical_jn(n, z):
    return 5.5

def eval_legendre(n, x):
    return 6.5

def roots_legendre(n):
    return ([-1.0, 1.0], [1.0, 1.0])

def multigammaln(a, d):
    return 7.5

def chdtr(v, x):
    return 8.5

def __getattr__(name):
    def _placeholder(*_args, **_kwargs):
        raise RuntimeError(f"unexpected scipy.special access: {name}")
    return _placeholder
"#,
        )
        .expect("write fake scipy.special module");

        let fixture_path = root.join("fixture.json");
        let output_path = root.join("oracle.json");
        let fixture = serde_json::json!({
            "packet_id": "TEST-EXTENDED-SPECIAL-MAP",
            "family": "special_core",
            "cases": [
                {
                    "case_id": "hyp2f1-case",
                    "function": "hyp2f1",
                    "args": [1.0, 2.0, 3.0, 0.25]
                },
                {
                    "case_id": "wright-bessel-case",
                    "function": "wright_bessel",
                    "args": [1.0, 2.0, 0.5]
                },
                {
                    "case_id": "spherical-jn-case",
                    "function": "spherical_jn",
                    "args": [2.0, 0.75]
                },
                {
                    "case_id": "eval-legendre-case",
                    "function": "eval_legendre",
                    "args": [3.0, 0.125]
                },
                {
                    "case_id": "roots-legendre-case",
                    "function": "roots_legendre",
                    "args": [4.0]
                },
                {
                    "case_id": "multigammaln-case",
                    "function": "multigammaln",
                    "args": [2.5, 3.0]
                },
                {
                    "case_id": "chdtr-case",
                    "function": "chdtr",
                    "args": [4.0, 1.5]
                }
            ]
        });
        fs::write(
            &fixture_path,
            serde_json::to_vec_pretty(&fixture).expect("serialize extended special map fixture"),
        )
        .expect("write fixture");

        let script_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python_oracle/scipy_special_oracle.py");
        let status = Command::new("python3")
            .env("PYTHONPATH", &pythonpath)
            .arg(&script_path)
            .arg("--fixture")
            .arg(&fixture_path)
            .arg("--output")
            .arg(&output_path)
            .arg("--oracle-root")
            .arg("/tmp/nonexistent-oracle")
            .status()
            .expect("run scipy special oracle");
        assert!(status.success(), "oracle script should exit successfully");

        let payload: serde_json::Value =
            serde_json::from_slice(&fs::read(&output_path).expect("read oracle output"))
                .expect("parse oracle output");
        let outputs = payload["case_outputs"]
            .as_array()
            .expect("oracle outputs should be an array");
        assert_eq!(outputs.len(), 7, "expected all extended map cases to run");
        for output in outputs {
            assert_eq!(
                output["status"], "ok",
                "unexpected oracle output: {output:?}"
            );
        }
        assert_eq!(outputs[0]["result_kind"], "scalar");
        assert_eq!(outputs[0]["result"]["value"], 3.5);
        assert_eq!(outputs[1]["result_kind"], "scalar");
        assert_eq!(outputs[1]["result"]["value"], 4.5);
        assert_eq!(outputs[2]["result_kind"], "scalar");
        assert_eq!(outputs[2]["result"]["value"], 5.5);
        assert_eq!(outputs[3]["result_kind"], "scalar");
        assert_eq!(outputs[3]["result"]["value"], 6.5);
        assert_eq!(outputs[4]["result_kind"], "tuple");
        assert_eq!(
            outputs[4]["result"]["values"],
            serde_json::json!([[-1.0, 1.0], [1.0, 1.0]])
        );
        assert_eq!(outputs[5]["result_kind"], "scalar");
        assert_eq!(outputs[5]["result"]["value"], 7.5);
        assert_eq!(outputs[6]["result_kind"], "scalar");
        assert_eq!(outputs[6]["result"]["value"], 8.5);
    }

    #[test]
    fn scipy_special_oracle_supports_all_current_special_fixture_functions() {
        let unique = format!(
            "fsci-conformance-test-special-fixture-coverage-{}",
            super::now_unix_ms()
        );
        let root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&root).expect("create temp root");

        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/FSCI-P2C-006_special_core.json");
        let output_path = root.join("oracle.json");
        let script_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python_oracle/scipy_special_oracle.py");

        let output = Command::new("python3")
            .arg(&script_path)
            .arg("--fixture")
            .arg(&fixture_path)
            .arg("--output")
            .arg(&output_path)
            .arg("--oracle-root")
            .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../legacy_scipy_code/scipy"))
            .output()
            .expect("run scipy special oracle");
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("No module named") {
                eprintln!("skipping live special-fixture oracle coverage test: {stderr}");
                return;
            }
            assert!(
                output.status.success(),
                "oracle script should exit successfully\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                stderr
            );
        }

        let payload: serde_json::Value =
            serde_json::from_slice(&fs::read(&output_path).expect("read oracle output"))
                .expect("parse oracle output");
        let outputs = payload["case_outputs"]
            .as_array()
            .expect("oracle outputs should be an array");
        let unsupported: Vec<String> = outputs
            .iter()
            .filter(|output| output["result_kind"] == "unsupported_function")
            .map(|output| {
                format!(
                    "{}: {}",
                    output["case_id"].as_str().unwrap_or("<missing-case-id>"),
                    output["error"].as_str().unwrap_or("<missing-error>")
                )
            })
            .collect();
        assert!(
            unsupported.is_empty(),
            "fixture still contains unsupported oracle functions: {unsupported:?}"
        );
    }

    #[test]
    fn scipy_special_oracle_btdtrc_uses_symmetry_to_preserve_near_one_precision() {
        let unique = format!(
            "fsci-conformance-test-btdtrc-symmetry-{}",
            super::now_unix_ms()
        );
        let root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&root).expect("create temp root");
        let pythonpath = root.join("pythonpath");
        let fake_scipy = pythonpath.join("scipy");
        fs::create_dir_all(&fake_scipy).expect("create fake scipy package");

        fs::write(
            pythonpath.join("numpy.py"),
            r#"__version__ = "mock-numpy-1.0"

class _ErrState:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False

def asarray(value, dtype=None):
    if dtype is float:
        return float(value)
    return value

nan = float("nan")

def isscalar(value):
    return not isinstance(value, (list, tuple, dict))

def iscomplexobj(value):
    if isinstance(value, (list, tuple)):
        return any(iscomplexobj(item) for item in value)
    return isinstance(value, complex)

def where(condition, when_true, when_false):
    return when_true if condition else when_false

def log1p(value):
    return value

def expm1(value):
    return value

def sinc(value):
    return value

def errstate(**_kwargs):
    return _ErrState()

def __getattr__(name):
    def _placeholder(*args, **_kwargs):
        if args:
            return args[0]
        raise RuntimeError(f"unexpected numpy access: {name}")
    return _placeholder
"#,
        )
        .expect("write fake numpy module");
        fs::write(
            fake_scipy.join("__init__.py"),
            r#"from . import special

__version__ = "mock-scipy-1.0"
"#,
        )
        .expect("write fake scipy package");
        fs::write(
            fake_scipy.join("special.py"),
            r#"def _close(a, b, tol=1e-12):
    return abs(a - b) <= tol

def betainc(a, b, x):
    if _close(a, 2.0) and _close(b, 3.0) and _close(x, 0.999999):
        return 1.0 - 4e-18
    if _close(a, 3.0) and _close(b, 2.0) and _close(x, 1e-6):
        return 4e-18
    raise RuntimeError(f"unexpected betainc arguments: {(a, b, x)!r}")

def __getattr__(name):
    def _placeholder(*_args, **_kwargs):
        raise RuntimeError(f"unexpected scipy.special access: {name}")
    return _placeholder
"#,
        )
        .expect("write fake scipy.special module");

        let fixture_path = root.join("fixture.json");
        let output_path = root.join("oracle.json");
        let fixture = serde_json::json!({
            "packet_id": "TEST-BTDTRC-SYMMETRY",
            "family": "special_core",
            "cases": [
                {
                    "case_id": "btdtrc-near-one",
                    "function": "btdtrc",
                    "args": [2.0, 3.0, 0.999999]
                }
            ]
        });
        fs::write(
            &fixture_path,
            serde_json::to_vec_pretty(&fixture).expect("serialize btdtrc fixture"),
        )
        .expect("write fixture");

        let script_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python_oracle/scipy_special_oracle.py");
        let status = Command::new("python3")
            .env("PYTHONPATH", &pythonpath)
            .arg(&script_path)
            .arg("--fixture")
            .arg(&fixture_path)
            .arg("--output")
            .arg(&output_path)
            .arg("--oracle-root")
            .arg("/tmp/nonexistent-oracle")
            .status()
            .expect("run scipy special oracle");
        assert!(status.success(), "oracle script should exit successfully");

        let payload: serde_json::Value =
            serde_json::from_slice(&fs::read(&output_path).expect("read oracle output"))
                .expect("parse oracle output");
        let output = &payload["case_outputs"][0];
        assert_eq!(output["status"], "ok");
        assert_eq!(output["result_kind"], "scalar");
        let value = output["result"]["value"]
            .as_f64()
            .expect("btdtrc result should be an f64");
        assert!(value > 0.0, "expected non-zero tail, got {value}");
        assert!(
            (value - 4e-18).abs() <= 1e-30,
            "expected symmetry-preserved tail, got {value}"
        );
    }

    #[test]
    fn scipy_special_oracle_rel_gamma_recurrence_accepts_complex_args() {
        let unique = format!(
            "fsci-conformance-test-runtime-error-{}",
            super::now_unix_ms()
        );
        let root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&root).expect("create temp root");
        let pythonpath = root.join("pythonpath");
        let fake_scipy = pythonpath.join("scipy");
        fs::create_dir_all(&fake_scipy).expect("create fake scipy package");

        fs::write(
            pythonpath.join("numpy.py"),
            r#"__version__ = "mock-numpy-1.0"

class _ErrState:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False

def asarray(value, dtype=None):
    if dtype is float:
        return float(value)
    return value

nan = float("nan")

def isscalar(value):
    return not isinstance(value, (list, tuple, dict))

def iscomplexobj(value):
    if isinstance(value, (list, tuple)):
        return any(iscomplexobj(item) for item in value)
    return isinstance(value, complex)

def where(condition, when_true, when_false):
    return when_true if condition else when_false

def log1p(value):
    return value

def expm1(value):
    return value

def sinc(value):
    return value

def errstate(**_kwargs):
    return _ErrState()

def logaddexp(a, b):
    import math as _m
    return _m.log(_m.exp(a) + _m.exp(b))

def logaddexp2(a, b):
    return logaddexp(a, b) / 0.6931471805599453
"#,
        )
        .expect("write fake numpy module");
        fs::write(
            fake_scipy.join("__init__.py"),
            r#"from . import special

__version__ = "mock-scipy-1.0"
"#,
        )
        .expect("write fake scipy package");
        fs::write(
            fake_scipy.join("special.py"),
            r#"def gamma(value):
    if value == complex(2.0, 1.0):
        return complex(5.0, -3.0)
    if value == complex(3.0, 1.0):
        return complex(2.0, 1.0) * complex(5.0, -3.0)
    raise RuntimeError(f"unexpected gamma argument: {value!r}")

def __getattr__(name):
    def _placeholder(*_args, **_kwargs):
        raise RuntimeError(f"unexpected scipy.special access: {name}")
    return _placeholder
"#,
        )
        .expect("write fake scipy.special module");

        let fixture_path = root.join("fixture.json");
        let output_path = root.join("oracle.json");
        let fixture = serde_json::json!({
            "packet_id": "TEST-REL-GAMMA",
            "family": "special_core",
            "cases": [
                {
                    "case_id": "rel-gamma-complex",
                    "function": "rel_gamma_recurrence",
                    "args": [
                        {
                            "re": 2.0,
                            "im": 1.0
                        }
                    ]
                }
            ]
        });
        fs::write(
            &fixture_path,
            serde_json::to_vec_pretty(&fixture).expect("serialize rel_gamma fixture"),
        )
        .expect("write fixture");

        let script_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python_oracle/scipy_special_oracle.py");
        let status = Command::new("python3")
            .env("PYTHONPATH", &pythonpath)
            .arg(&script_path)
            .arg("--fixture")
            .arg(&fixture_path)
            .arg("--output")
            .arg(&output_path)
            .arg("--oracle-root")
            .arg("/tmp/nonexistent-oracle")
            .status()
            .expect("run scipy special oracle");
        assert!(status.success(), "oracle script should exit successfully");

        let payload: serde_json::Value =
            serde_json::from_slice(&fs::read(&output_path).expect("read oracle output"))
                .expect("parse oracle output");
        let output = &payload["case_outputs"][0];
        assert_eq!(output["case_id"], "rel-gamma-complex");
        assert_eq!(output["status"], "ok");
        assert_eq!(output["result_kind"], "complex_scalar");
        let value = &output["result"]["value"];
        assert!(
            value["re"].as_f64().expect("real component").abs() < 1.0e-12,
            "expected recurrence residual near zero, got {output:?}"
        );
        assert!(
            value["im"].as_f64().expect("imag component").abs() < 1.0e-12,
            "expected recurrence residual near zero, got {output:?}"
        );
    }

    #[test]
    fn scipy_special_oracle_rel_jn_recurrence_accepts_complex_args() {
        let unique = format!("fsci-conformance-test-rel-jn-cplx-{}", super::now_unix_ms());
        let root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&root).expect("create temp root");
        let pythonpath = root.join("pythonpath");
        let fake_scipy = pythonpath.join("scipy");
        fs::create_dir_all(&fake_scipy).expect("create fake scipy package");

        fs::write(
            pythonpath.join("numpy.py"),
            r#"__version__ = "mock-numpy-1.0"

class _ErrState:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False

def asarray(value, dtype=None):
    if dtype is float:
        return float(value)
    return value

nan = float("nan")

def isscalar(value):
    return not isinstance(value, (list, tuple, dict))

def iscomplexobj(value):
    if isinstance(value, (list, tuple)):
        return any(iscomplexobj(item) for item in value)
    return isinstance(value, complex)

def where(condition, when_true, when_false):
    return when_true if condition else when_false

def log1p(value):
    return value

def expm1(value):
    return value

def sinc(value):
    return value

def errstate(**_kwargs):
    return _ErrState()

def logaddexp(a, b):
    import math as _m
    return _m.log(_m.exp(a) + _m.exp(b))

def logaddexp2(a, b):
    return logaddexp(a, b) / 0.6931471805599453
"#,
        )
        .expect("write fake numpy module");
        fs::write(
            fake_scipy.join("__init__.py"),
            r#"from . import special

__version__ = "mock-scipy-1.0"
"#,
        )
        .expect("write fake scipy package");
        fs::write(
            fake_scipy.join("special.py"),
            r#"def jv(order, x):
    if order == 1.0:
        return complex(1.0, 0.0)
    if order == 2.0:
        return x / 4.0
    if order == 3.0:
        return complex(0.0, 0.0)
    raise RuntimeError(f"unexpected jv order: {order!r}")

def __getattr__(name):
    def _placeholder(*_args, **_kwargs):
        raise RuntimeError(f"unexpected scipy.special access: {name}")
    return _placeholder
"#,
        )
        .expect("write fake scipy.special module");

        let fixture_path = root.join("fixture.json");
        let output_path = root.join("oracle.json");
        let fixture = serde_json::json!({
            "packet_id": "TEST-REL-JN",
            "family": "special_core",
            "cases": [
                {
                    "case_id": "rel-jn-complex",
                    "function": "rel_jn_recurrence",
                    "args": [
                        2.0,
                        {
                            "re": 1.25,
                            "im": 0.75
                        }
                    ]
                }
            ]
        });
        fs::write(
            &fixture_path,
            serde_json::to_vec_pretty(&fixture).expect("serialize rel_jn fixture"),
        )
        .expect("write fixture");

        let script_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python_oracle/scipy_special_oracle.py");
        let status = Command::new("python3")
            .env("PYTHONPATH", &pythonpath)
            .arg(&script_path)
            .arg("--fixture")
            .arg(&fixture_path)
            .arg("--output")
            .arg(&output_path)
            .arg("--oracle-root")
            .arg("/tmp/nonexistent-oracle")
            .status()
            .expect("run scipy special oracle");
        assert!(status.success(), "oracle script should exit successfully");

        let payload: serde_json::Value =
            serde_json::from_slice(&fs::read(&output_path).expect("read oracle output"))
                .expect("parse oracle output");
        let output = &payload["case_outputs"][0];
        assert_eq!(output["case_id"], "rel-jn-complex");
        assert_eq!(output["status"], "ok");
        assert_eq!(output["result_kind"], "complex_scalar");
        let value = &output["result"]["value"];
        assert!(
            value["re"].as_f64().expect("real component").abs() < 1.0e-12,
            "expected recurrence residual near zero, got {output:?}"
        );
        assert!(
            value["im"].as_f64().expect("imag component").abs() < 1.0e-12,
            "expected recurrence residual near zero, got {output:?}"
        );
    }

    #[test]
    fn scipy_special_oracle_unexpected_runtime_error_propagates() {
        let unique = format!("fsci-conformance-test-runtime-err-{}", super::now_unix_ms());
        let root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&root).expect("create temp root");
        let pythonpath = root.join("pythonpath");
        let fake_scipy = pythonpath.join("scipy");
        fs::create_dir_all(&fake_scipy).expect("create fake scipy package");

        fs::write(
            pythonpath.join("numpy.py"),
            r#"__version__ = "mock-numpy-1.0"

def asarray(value, dtype=None):
    return value

def log1p(value):
    return value

def expm1(value):
    return value

def sinc(value):
    return value

def isscalar(value):
    return not isinstance(value, (list, tuple, dict))

def iscomplexobj(value):
    return isinstance(value, complex)
"#,
        )
        .expect("write fake numpy module");
        fs::write(
            fake_scipy.join("__init__.py"),
            r#"from . import special

__version__ = "mock-scipy-1.0"
"#,
        )
        .expect("write fake scipy package");
        fs::write(
            fake_scipy.join("special.py"),
            r#"def gamma(*_args, **_kwargs):
    raise RuntimeError("boom")

def __getattr__(name):
    def _placeholder(*_args, **_kwargs):
        raise RuntimeError(f"unexpected scipy.special access: {name}")
    return _placeholder
"#,
        )
        .expect("write fake scipy.special module");

        let fixture_path = root.join("fixture.json");
        let output_path = root.join("oracle.json");
        let fixture = serde_json::json!({
            "packet_id": "TEST-RUNTIME-ERROR",
            "family": "special_core",
            "cases": [
                {
                    "case_id": "gamma-runtime-error",
                    "function": "gamma",
                    "args": [1.0]
                }
            ]
        });
        fs::write(
            &fixture_path,
            serde_json::to_vec_pretty(&fixture).expect("serialize runtime error fixture"),
        )
        .expect("write fixture");

        let script_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python_oracle/scipy_special_oracle.py");
        let status = Command::new("python3")
            .env("PYTHONPATH", &pythonpath)
            .arg(&script_path)
            .arg("--fixture")
            .arg(&fixture_path)
            .arg("--output")
            .arg(&output_path)
            .arg("--oracle-root")
            .arg("/tmp/nonexistent-oracle")
            .status()
            .expect("run scipy special oracle");
        assert!(
            !status.success(),
            "unexpected runtime errors must propagate and fail the oracle script"
        );
    }

    #[test]
    fn run_linalg_packet_with_mock_oracle_uses_oracle_values() {
        let unique = format!("fsci-conformance-mock-run-{}", super::now_unix_ms());
        let root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&root).expect("create temp root");
        let fixtures = root.join("fixtures");
        fs::create_dir_all(&fixtures).expect("create fixtures");

        let fixture_dst = fixtures.join("FSCI-P2C-002_linalg_core.json");
        let fixture = LinalgPacketFixture {
            packet_id: "FSCI-P2C-002".to_owned(),
            family: "linalg_core".to_owned(),
            cases: vec![LinalgCase::Solve {
                case_id: "solve_general_2x2".to_owned(),
                mode: RuntimeMode::Strict,
                a: vec![vec![3.0, 2.0], vec![1.0, 2.0]],
                b: vec![5.0, 5.0],
                assume_a: None,
                lower: None,
                transposed: None,
                check_finite: None,
                expected: LinalgExpectedOutcome::Vector {
                    values: vec![999.0, -999.0],
                    atol: 1e-12,
                    rtol: 1e-12,
                    expect_warning_ill_conditioned: None,
                },
            }],
            oracle_provenance: None,
        };
        fs::write(
            &fixture_dst,
            serde_json::to_vec_pretty(&fixture).expect("serialize fixture"),
        )
        .expect("write fixture");

        let script_path = root.join("mock_oracle.py");
        let script = r#"
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--fixture", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--oracle-root", required=True)
args = parser.parse_args()

fixture = json.loads(Path(args.fixture).read_text())
result = {
    "packet_id": fixture["packet_id"],
    "family": fixture["family"],
    "generated_unix_ms": 0,
    "runtime": {
        "python_version": "3.11.0",
        "numpy_version": "2.0.0",
        "scipy_version": "mock-1.0",
    },
    "case_outputs": [
        {
            "case_id": "solve_general_2x2",
            "status": "ok",
            "result_kind": "vector",
            "result": {"values": [0.0, 2.5]},
            "error": None,
        }
    ],
}
Path(args.output).write_text(json.dumps(result, indent=2))
"#;
        fs::write(&script_path, script).expect("write mock script");

        let cfg = HarnessConfig {
            oracle_root: PathBuf::from("/tmp/nonexistent-oracle"),
            fixture_root: fixtures,
            strict_mode: true,
        };
        let oracle = PythonOracleConfig {
            python_bin: PathBuf::from("python3"),
            script_path,
            required: true,
        };

        let report =
            run_linalg_packet_with_oracle_capture(&cfg, "FSCI-P2C-002_linalg_core.json", &oracle)
                .expect("oracle-backed linalg packet runs");
        assert_eq!(report.failed_cases, 0);
        assert_eq!(report.passed_cases, 1);

        let capture_path = cfg
            .artifact_dir_for("FSCI-P2C-002")
            .join("oracle_capture.json");
        let capture = load_oracle_capture(&capture_path).expect("load captured oracle output");
        assert_eq!(capture.case_outputs.len(), 1);
        assert!(capture.provenance.is_some());
    }

    #[test]
    fn scipy_oracle_capture_when_available() {
        let scipy_check = Command::new("python3")
            .arg("-c")
            .arg("import scipy")
            .status();
        if !matches!(scipy_check, Ok(status) if status.success()) {
            eprintln!("SciPy not available in this environment; skipping optional oracle test");
            return;
        }

        let unique = format!("fsci-conformance-scipy-{}", super::now_unix_ms());
        let root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&root).expect("create temp root");
        let fixtures = root.join("fixtures");
        fs::create_dir_all(&fixtures).expect("create fixtures");

        let fixture_src = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-002_linalg_core.json");
        let fixture_dst = fixtures.join("FSCI-P2C-002_linalg_core.json");
        fs::copy(&fixture_src, &fixture_dst).expect("copy fixture");

        let cfg = HarnessConfig {
            oracle_root: HarnessConfig::default_paths().oracle_root,
            fixture_root: fixtures,
            strict_mode: true,
        };

        let oracle = PythonOracleConfig {
            required: true,
            script_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("python_oracle/scipy_linalg_oracle.py"),
            ..PythonOracleConfig::default()
        };

        let report =
            run_linalg_packet_with_oracle_capture(&cfg, "FSCI-P2C-002_linalg_core.json", &oracle)
                .expect("scipy-backed linalg packet runs");
        assert_eq!(report.failed_cases, 0);
        let output_path = cfg
            .artifact_dir_for("FSCI-P2C-002")
            .join("oracle_capture.json");
        let parsed = load_oracle_capture(&output_path).expect("oracle capture parse succeeds");

        let fixture_raw = fs::read_to_string(fixture_dst).expect("read fixture");
        let fixture: LinalgPacketFixture =
            serde_json::from_str(&fixture_raw).expect("fixture parse");
        assert_eq!(parsed.packet_id, "FSCI-P2C-002");
        assert_eq!(parsed.case_outputs.len(), fixture.cases.len());
        assert!(
            parsed
                .provenance
                .as_ref()
                .expect("scipy capture should include provenance")
                .capture_blake3
                .len()
                >= 10
        );
    }

    #[test]
    fn run_differential_stats_with_mock_oracle_uses_oracle_values() {
        let unique = format!("fsci-conformance-stats-mock-run-{}", super::now_unix_ms());
        let root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&root).expect("create temp root");
        let fixtures = root.join("fixtures");
        fs::create_dir_all(&fixtures).expect("create fixtures");

        let fixture_dst = fixtures.join("FSCI-P2C-012_stats_core.json");
        let fixture = StatsPacketFixture {
            packet_id: "FSCI-P2C-012".to_owned(),
            family: "stats_core".to_owned(),
            cases: vec![StatsCase {
                case_id: "sem_basic".to_owned(),
                category: "differential".to_owned(),
                mode: RuntimeMode::Strict,
                function: "sem".to_owned(),
                args: vec![serde_json::json!([1.0, 2.0, 3.0, 4.0])],
                expected: StatsExpected {
                    kind: "scalar".to_owned(),
                    value: Some(999.0),
                    nobs: None,
                    minmax: None,
                    mean: None,
                    variance: None,
                    skewness: None,
                    kurtosis: None,
                    statistic: None,
                    pvalue: None,
                    slope: None,
                    intercept: None,
                    rvalue: None,
                    stderr: None,
                    array_value: None,
                    atol: Some(1e-12),
                    rtol: Some(1e-12),
                    contract_ref: "sem".to_owned(),
                },
            }],
        };
        fs::write(
            &fixture_dst,
            serde_json::to_vec_pretty(&fixture).expect("serialize fixture"),
        )
        .expect("write fixture");

        let script_path = root.join("mock_stats_oracle.py");
        let script = r#"
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--fixture", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--oracle-root", required=True)
args = parser.parse_args()

fixture = json.loads(Path(args.fixture).read_text())
result = {
    "packet_id": fixture["packet_id"],
    "family": fixture["family"],
    "generated_unix_ms": 0,
    "runtime": {
        "python_version": "3.11.0",
        "numpy_version": "2.0.0",
        "scipy_version": "mock-1.0",
    },
    "case_outputs": [
        {
            "case_id": "sem_basic",
            "status": "ok",
            "result_kind": "scalar",
            "result": {"value": 0.6454972243679028},
            "error": None,
        }
    ],
}
Path(args.output).write_text(json.dumps(result, indent=2))
"#;
        fs::write(&script_path, script).expect("write mock script");

        let oracle = DifferentialOracleConfig {
            python_path: PathBuf::from("python3"),
            script_path,
            timeout_secs: 30,
            required: true,
        };

        let report =
            run_differential_test(&fixture_dst, &oracle).expect("oracle-backed stats fixture runs");
        assert_eq!(report.fail_count, 0);
        assert_eq!(report.pass_count, 1);
        assert_eq!(report.oracle_status, OracleStatus::Available);
        assert!(
            report.per_case_results[0].message.contains("scalar match"),
            "unexpected message: {}",
            report.per_case_results[0].message
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Differential conformance harness tests (bd-3jh.4)
    // ═══════════════════════════════════════════════════════════════

    fn default_test_oracle() -> DifferentialOracleConfig {
        DifferentialOracleConfig {
            required: false,
            ..DifferentialOracleConfig::default()
        }
    }

    #[derive(Debug, Serialize)]
    struct StructuredCaseLog {
        test_id: String,
        category: String,
        input_summary: String,
        expected: String,
        actual: String,
        diff: Option<f64>,
        tolerance: Option<super::ToleranceUsed>,
        pass: bool,
    }

    fn optimize_case_input_summary(case: &super::OptimizeCase) -> String {
        match case {
            super::OptimizeCase::Minimize {
                method,
                objective,
                x0,
                hessp,
                ..
            } => format!(
                "operation=minimize method={method:?} objective={objective:?} x0={x0:?} hessp={hessp}"
            ),
            super::OptimizeCase::Root {
                method,
                objective,
                bracket,
                ..
            } => format!(
                "operation=root method={method:?} objective={objective:?} bracket={bracket:?}"
            ),
            super::OptimizeCase::DifferentialEvolution {
                objective,
                bounds,
                seed,
                ..
            } => format!(
                "operation=differential_evolution objective={objective:?} bounds={bounds:?} seed={seed:?}"
            ),
            super::OptimizeCase::Basinhopping {
                objective,
                x0,
                seed,
                ..
            } => format!("operation=basinhopping objective={objective:?} x0={x0:?} seed={seed:?}"),
            super::OptimizeCase::DualAnnealing {
                objective,
                bounds,
                seed,
                ..
            } => format!(
                "operation=dual_annealing objective={objective:?} bounds={bounds:?} seed={seed:?}"
            ),
            super::OptimizeCase::Brute {
                objective,
                ranges,
                ns,
                ..
            } => format!("operation=brute objective={objective:?} ranges={ranges:?} ns={ns:?}"),
        }
    }

    fn optimize_case_expected_summary(expected: &super::OptimizeExpectedOutcome) -> String {
        match expected {
            super::OptimizeExpectedOutcome::MinimizePoint { x, fun, .. } => {
                format!("minimize_point x={x:?} fun={fun:.6e}")
            }
            super::OptimizeExpectedOutcome::RootValue { root, .. } => {
                format!("root_value root={root:.6e}")
            }
            super::OptimizeExpectedOutcome::MinimizeStatus { status, success } => {
                format!("minimize_status status={status:?} success={success}")
            }
            super::OptimizeExpectedOutcome::RootStatus { status, converged } => {
                format!("root_status status={status:?} converged={converged}")
            }
            super::OptimizeExpectedOutcome::Error { error } => format!("error={error}"),
        }
    }

    fn special_case_input_summary(case: &super::SpecialCase) -> String {
        format!(
            "function={:?} mode={:?} args={:?}",
            case.function, case.mode, case.args
        )
    }

    fn special_case_expected_summary(expected: &super::SpecialExpectedOutcome) -> String {
        match expected {
            super::SpecialExpectedOutcome::Scalar {
                value,
                atol,
                rtol,
                contract_ref,
            } => format!(
                "scalar value={value:.16e} atol={atol:?} rtol={rtol:?} contract_ref={contract_ref:?}"
            ),
            super::SpecialExpectedOutcome::Vector {
                values,
                atol,
                rtol,
                contract_ref,
            } => format!(
                "vector values={values:?} atol={atol:?} rtol={rtol:?} contract_ref={contract_ref:?}"
            ),
            super::SpecialExpectedOutcome::ComplexScalar {
                value,
                atol,
                rtol,
                contract_ref,
            } => format!(
                "complex_scalar value=({}, {}) atol={atol:?} rtol={rtol:?} contract_ref={contract_ref:?}",
                value.re, value.im
            ),
            super::SpecialExpectedOutcome::ComplexVector {
                values,
                atol,
                rtol,
                contract_ref,
            } => format!(
                "complex_vector values={values:?} atol={atol:?} rtol={rtol:?} contract_ref={contract_ref:?}"
            ),
            super::SpecialExpectedOutcome::Class { class } => {
                format!("class={class:?}")
            }
            super::SpecialExpectedOutcome::ErrorKind { error } => {
                format!("error_kind={error}")
            }
        }
    }

    fn array_api_case_input_summary(case: &super::ArrayApiCase) -> String {
        format!("{case:?}")
    }

    fn array_api_case_expected_summary(expected: &super::ArrayApiExpectedOutcome) -> String {
        format!("{expected:?}")
    }

    fn integrate_case_input_summary(case: &super::IntegrateCase) -> String {
        format!(
            "function={} mode={:?} args={:?}",
            case.function, case.mode, case.args
        )
    }

    fn integrate_case_expected_summary(expected: &super::IntegrateExpected) -> String {
        format!(
            "kind={} value={:?} atol={:?} rtol={:?} contract_ref={}",
            expected.kind, expected.value, expected.atol, expected.rtol, expected.contract_ref
        )
    }

    fn stats_case_input_summary(case: &super::StatsCase) -> String {
        format!(
            "function={} mode={:?} args={:?}",
            case.function, case.mode, case.args
        )
    }

    fn stats_case_expected_summary(expected: &super::StatsExpected) -> String {
        format!("{expected:?}")
    }

    fn signal_case_input_summary(case: &super::SignalCase) -> String {
        format!(
            "function={} mode={:?} args={:?}",
            case.function, case.mode, case.args
        )
    }

    fn signal_case_expected_summary(expected: &super::SignalExpected) -> String {
        format!("{expected:?}")
    }

    fn spatial_case_input_summary(case: &super::SpatialCase) -> String {
        format!(
            "function={} mode={:?} args={:?}",
            case.function, case.mode, case.args
        )
    }

    fn spatial_case_expected_summary(expected: &super::SpatialExpected) -> String {
        format!("{expected:?}")
    }

    fn cluster_case_input_summary(case: &super::ClusterCase) -> String {
        format!(
            "function={} mode={:?} seed={:?} args={:?}",
            case.function, case.mode, case.seed, case.args
        )
    }

    fn cluster_case_expected_summary(expected: &super::ClusterExpected) -> String {
        format!("{expected:?}")
    }

    fn casp_case_input_summary(case: &super::CaspCase) -> String {
        format!(
            "kind={:?} mode={:?} condition_signal={:?} metadata_signal={:?} anomaly_signal={:?} condition_state={:?} observations={:?}",
            case.test_kind,
            case.mode,
            case.condition_signal,
            case.metadata_signal,
            case.anomaly_signal,
            case.condition_state,
            case.observations
        )
    }

    fn casp_case_expected_summary(expected: &super::CaspExpectedOutcome) -> String {
        format!("{expected:?}")
    }

    fn fft_case_input_summary(case: &super::FftCase) -> String {
        format!(
            "transform={:?} mode={:?} normalization={:?} real_input={:?} complex_input={:?} output_len={:?} sample_spacing={:?} shape={:?}",
            case.transform,
            case.mode,
            case.normalization,
            case.real_input,
            case.complex_input,
            case.output_len,
            case.sample_spacing,
            case.shape
        )
    }

    fn fft_case_expected_summary(expected: &super::FftExpectedOutcome) -> String {
        format!("{expected:?}")
    }

    #[test]
    fn differential_test_validate_tol_fixture() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-001_validate_tol.json");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle)
            .expect("differential validate_tol should succeed");

        assert_eq!(report.packet_id, "FSCI-P2C-001");
        assert_eq!(report.family, "integrate.validate_tol");
        assert_eq!(report.fail_count, 0);
        assert!(report.pass_count >= 1);
        assert_eq!(
            report.per_case_results.len(),
            report.pass_count + report.fail_count
        );

        let audit_path =
            super::differential_audit_ledger_path_for_fixture(&fixture_path, &report.packet_id);
        let audit_jsonl = fs::read_to_string(&audit_path).expect("read validate_tol audit ledger");
        assert!(audit_path.exists());
        assert!(audit_jsonl.lines().count() >= 2);
        assert!(audit_jsonl.contains("\"kind\":\"fail_closed\""));
    }

    #[test]
    fn differential_test_validate_tol_hardened_clamp_emits_bounded_recovery() {
        let unique = format!("fsci-conformance-audit-{}", super::now_unix_ms());
        let root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&root).expect("create temp root");

        let fixture_path = root.join("FSCI-P2C-001_validate_tol_audit.json");
        let fixture = super::PacketFixture {
            packet_id: "FSCI-P2C-001-AUDIT".to_owned(),
            family: "integrate.validate_tol".to_owned(),
            cases: vec![super::ValidateTolCase {
                case_id: "hardened_clamp".to_owned(),
                mode: RuntimeMode::Hardened,
                n: 1,
                rtol: super::FixtureToleranceValue::Scalar(1.0e-16),
                atol: super::FixtureToleranceValue::Scalar(1.0e-8),
                expected: super::ExpectedOutcome::Ok {
                    rtol: super::FixtureToleranceValue::Scalar(2.220446049250313e-14),
                    atol: super::FixtureToleranceValue::Scalar(1.0e-8),
                    warning_rtol_clamped: true,
                },
            }],
        };
        fs::write(
            &fixture_path,
            serde_json::to_vec_pretty(&fixture).expect("serialize fixture"),
        )
        .expect("write fixture");

        let report =
            run_differential_test(&fixture_path, &default_test_oracle()).expect("fixture runs");
        assert_eq!(report.fail_count, 0);

        let audit_path =
            super::differential_audit_ledger_path_for_fixture(&fixture_path, &report.packet_id);
        let audit_jsonl =
            fs::read_to_string(&audit_path).expect("read hardened clamp audit ledger");
        assert!(audit_path.exists());
        assert!(audit_jsonl.contains("\"kind\":\"bounded_recovery\""));
    }

    #[test]
    fn differential_test_linalg_fixture() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-002_linalg_core.json");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle)
            .expect("differential linalg should succeed");

        assert_eq!(report.packet_id, "FSCI-P2C-002");
        assert_eq!(report.family, "linalg_core");
        assert_eq!(report.fail_count, 0);
        assert!(report.pass_count >= 1);

        // Linalg cases should produce max_diff values
        for case in &report.per_case_results {
            assert!(
                case.passed,
                "case {} should pass: {}",
                case.case_id, case.message
            );
            // Numeric cases have max_diff; error cases don't
            if case.tolerance_used.is_some() {
                assert!(
                    case.max_diff.is_some(),
                    "numeric case {} should have max_diff",
                    case.case_id
                );
            }
        }

        let audit_path =
            super::differential_audit_ledger_path_for_fixture(&fixture_path, &report.packet_id);
        let audit_jsonl = fs::read_to_string(&audit_path).expect("read linalg audit ledger");
        assert!(audit_path.exists());
        assert!(!audit_jsonl.trim().is_empty());
        assert!(audit_jsonl.contains("\"kind\":\"mode_decision\""));
        assert!(audit_jsonl.contains("\"kind\":\"fail_closed\""));
    }

    #[test]
    fn differential_linalg_audit_records_non_solve_case() {
        let case = LinalgCase::Det {
            case_id: "det_non_solve_audit".to_string(),
            mode: RuntimeMode::Strict,
            a: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            check_finite: Some(true),
            expected: LinalgExpectedOutcome::Scalar {
                value: -2.0,
                atol: 1.0e-12,
                rtol: 1.0e-12,
            },
        };
        let audit_ledger = fsci_linalg::sync_audit_ledger();

        let observed =
            super::execute_linalg_case_with_differential_audit(&case, &audit_ledger).expect("det");
        assert!(matches!(
            observed,
            super::LinalgObservedOutcome::Scalar { value } if (value + 2.0).abs() < 1.0e-12
        ));

        let ledger = audit_ledger.lock().expect("lock");
        assert_eq!(ledger.len(), 1);
        assert!(
            ledger.entries().iter().any(|entry| matches!(
                entry.action,
                fsci_runtime::AuditAction::ModeDecision { .. }
            ))
        );
    }

    #[test]
    fn optimize_packet_runner_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_optimize_packet(&cfg, "FSCI-P2C-003_optimize_core.json")
            .expect("optimize packet fixture should run");
        assert_eq!(
            report.failed_cases,
            0,
            "{}",
            serde_json::to_string(&report).unwrap()
        );
        assert!(
            report.passed_cases >= 29,
            "expected full optimize fixture execution"
        );

        let artifacts = write_parity_artifacts(&cfg, &report)
            .expect("optimize parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn p2c003_optimize_global_local_parity() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-003_optimize_core.json");
        let raw = fs::read_to_string(&fixture_path).expect("read optimize fixture");
        let fixture: OptimizePacketFixture =
            serde_json::from_str(&raw).expect("parse optimize fixture");

        let mut local_pairs = std::collections::BTreeSet::new();
        let mut global_ops = std::collections::BTreeSet::new();
        let mut global_objectives = std::collections::BTreeSet::new();
        let mut benchmark_differential_cases = 0usize;

        for case in &fixture.cases {
            match case {
                super::OptimizeCase::Minimize {
                    category,
                    method,
                    objective,
                    expected,
                    ..
                } if category == "differential" => {
                    let pair = format!("{method:?}:{objective:?}");
                    match method {
                        fsci_opt::OptimizeMethod::Bfgs
                        | fsci_opt::OptimizeMethod::ConjugateGradient
                        | fsci_opt::OptimizeMethod::Powell => {
                            local_pairs.insert(pair);
                            benchmark_differential_cases += 1;
                            assert!(
                                matches!(
                                    expected,
                                    super::OptimizeExpectedOutcome::MinimizePoint { .. }
                                ),
                                "local benchmark case {} must compare a minimizer point",
                                case.case_id()
                            );
                        }
                        _ => {}
                    }
                }
                super::OptimizeCase::DifferentialEvolution {
                    category,
                    objective,
                    expected,
                    ..
                }
                | super::OptimizeCase::Basinhopping {
                    category,
                    objective,
                    expected,
                    ..
                }
                | super::OptimizeCase::DualAnnealing {
                    category,
                    objective,
                    expected,
                    ..
                }
                | super::OptimizeCase::Brute {
                    category,
                    objective,
                    expected,
                    ..
                } if category == "differential" => {
                    let operation = match case {
                        super::OptimizeCase::DifferentialEvolution { .. } => {
                            "differential_evolution"
                        }
                        super::OptimizeCase::Basinhopping { .. } => "basinhopping",
                        super::OptimizeCase::DualAnnealing { .. } => "dual_annealing",
                        super::OptimizeCase::Brute { .. } => "brute",
                        _ => {
                            assert!(
                                matches!(
                                    case,
                                    super::OptimizeCase::DifferentialEvolution { .. }
                                        | super::OptimizeCase::Basinhopping { .. }
                                        | super::OptimizeCase::DualAnnealing { .. }
                                        | super::OptimizeCase::Brute { .. }
                                ),
                                "matched global optimize case"
                            );
                            "unknown_global_optimizer"
                        }
                    };
                    global_ops.insert(operation);
                    global_objectives.insert(format!("{objective:?}"));
                    benchmark_differential_cases += 1;
                    assert!(
                        matches!(
                            expected,
                            super::OptimizeExpectedOutcome::MinimizePoint { .. }
                        ),
                        "global benchmark case {} must compare a minimizer point",
                        case.case_id()
                    );
                }
                _ => {}
            }
        }

        for pair in [
            "Bfgs:Rosenbrock2",
            "Bfgs:Ackley2",
            "Bfgs:Rastrigin2",
            "ConjugateGradient:Rosenbrock2",
            "ConjugateGradient:Ackley2",
            "ConjugateGradient:Rastrigin2",
            "Powell:Rosenbrock2",
            "Powell:Ackley2",
            "Powell:Rastrigin2",
        ] {
            assert!(local_pairs.contains(pair), "missing local pair {pair}");
        }

        for operation in [
            "differential_evolution",
            "basinhopping",
            "dual_annealing",
            "brute",
        ] {
            assert!(
                global_ops.contains(operation),
                "missing global operation {operation}"
            );
        }

        for objective in [
            "Rosenbrock2",
            "Ackley2",
            "Rastrigin2",
            "Himmelblau2",
            "Sphere2",
        ] {
            assert!(
                global_objectives.contains(objective),
                "missing global benchmark objective {objective}"
            );
        }
        assert!(
            benchmark_differential_cases >= 12,
            "expected at least 12 local/global benchmark differential cases"
        );

        let oracle = default_test_oracle();
        let report =
            run_differential_test(&fixture_path, &oracle).expect("optimize differential runs");
        assert_eq!(report.fail_count, 0);
    }

    #[test]
    fn special_packet_runner_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_special_packet(&cfg, "FSCI-P2C-006_special_core.json")
            .expect("special packet fixture should run");
        assert_eq!(
            report.failed_cases,
            0,
            "{}",
            serde_json::to_string(&report).unwrap()
        );
        assert!(
            report.passed_cases >= 1,
            "expected at least one special function test case"
        );

        let artifacts = write_parity_artifacts(&cfg, &report)
            .expect("special parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn array_api_packet_runner_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_array_api_packet(&cfg, "FSCI-P2C-007_arrayapi_core.json")
            .expect("array_api packet fixture should run");
        assert_eq!(
            report.failed_cases,
            0,
            "{}",
            serde_json::to_string(&report).unwrap()
        );
        assert!(
            report.passed_cases >= 1,
            "expected at least one array_api test case"
        );

        let artifacts = write_parity_artifacts(&cfg, &report)
            .expect("array_api parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn sparse_packet_runner_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_sparse_packet(&cfg, "FSCI-P2C-004_sparse_ops.json")
            .expect("sparse packet fixture should run");
        assert_eq!(
            report.failed_cases,
            0,
            "{}",
            serde_json::to_string(&report).unwrap()
        );
        assert!(
            report.passed_cases >= 1,
            "expected at least one sparse test case"
        );

        let artifacts =
            write_parity_artifacts(&cfg, &report).expect("sparse parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn fft_packet_runner_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_fft_packet(&cfg, "FSCI-P2C-005_fft_core.json")
            .expect("fft packet fixture should run");
        assert_eq!(
            report.failed_cases,
            0,
            "{}",
            serde_json::to_string(&report).unwrap()
        );
        assert!(
            report.passed_cases >= 1,
            "expected at least one fft test case"
        );

        let artifacts =
            write_parity_artifacts(&cfg, &report).expect("fft parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn casp_packet_runner_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_casp_packet(&cfg, "FSCI-P2C-008_runtime_casp.json")
            .expect("casp packet fixture should run");
        assert_eq!(
            report.failed_cases,
            0,
            "{}",
            serde_json::to_string(&report).unwrap()
        );
        assert!(
            report.passed_cases >= 1,
            "expected at least one casp test case"
        );

        let artifacts =
            write_parity_artifacts(&cfg, &report).expect("casp parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn cluster_packet_runner_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_cluster_packet(&cfg, "FSCI-P2C-009_cluster_core.json")
            .expect("cluster packet fixture should run");
        assert_eq!(
            report.failed_cases,
            0,
            "{}",
            serde_json::to_string(&report).unwrap()
        );
        assert!(
            report.passed_cases >= 1,
            "expected at least one cluster test case"
        );

        let artifacts = write_parity_artifacts(&cfg, &report)
            .expect("cluster parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn spatial_packet_runner_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_spatial_packet(&cfg, "FSCI-P2C-010_spatial_core.json")
            .expect("spatial packet fixture should run");
        assert_eq!(
            report.failed_cases,
            0,
            "{}",
            serde_json::to_string(&report).unwrap()
        );
        assert!(
            report.passed_cases >= 1,
            "expected at least one spatial test case"
        );

        let artifacts = write_parity_artifacts(&cfg, &report)
            .expect("spatial parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn signal_packet_runner_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_signal_packet(&cfg, "FSCI-P2C-011_signal_core.json")
            .expect("signal packet fixture should run");
        assert_eq!(
            report.failed_cases,
            0,
            "{}",
            serde_json::to_string(&report).unwrap()
        );
        assert!(
            report.passed_cases >= 1,
            "expected at least one signal test case"
        );

        let artifacts =
            write_parity_artifacts(&cfg, &report).expect("signal parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn stats_packet_runner_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_stats_packet(&cfg, "FSCI-P2C-012_stats_core.json")
            .expect("stats packet fixture should run");
        assert_eq!(
            report.failed_cases,
            0,
            "{}",
            serde_json::to_string(&report).unwrap()
        );
        assert!(
            report.passed_cases >= 1,
            "expected at least one stats test case"
        );

        let artifacts =
            write_parity_artifacts(&cfg, &report).expect("stats parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn integrate_packet_runner_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_integrate_packet(&cfg, "FSCI-P2C-013_integrate_core.json")
            .expect("integrate packet fixture should run");
        assert_eq!(
            report.failed_cases,
            0,
            "{}",
            serde_json::to_string(&report).unwrap()
        );
        assert!(
            report.passed_cases >= 1,
            "expected at least one integrate test case"
        );

        let artifacts = write_parity_artifacts(&cfg, &report)
            .expect("integrate parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn interpolate_packet_runner_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_interpolate_packet(&cfg, "FSCI-P2C-014_interpolate_core.json")
            .expect("interpolate packet fixture should run");
        assert_eq!(
            report.failed_cases,
            0,
            "{}",
            serde_json::to_string(&report).unwrap()
        );
        assert_eq!(
            report.passed_cases, INTERPOLATE_CORE_CASE_COUNT,
            "interpolate packet should keep its full fixture surface"
        );

        let artifacts = write_parity_artifacts(&cfg, &report)
            .expect("interpolate parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn ndimage_packet_runner_passes() {
        let cfg = HarnessConfig::default_paths();
        let report = run_ndimage_packet(&cfg, "FSCI-P2C-015_ndimage_core.json")
            .expect("ndimage packet fixture should run");
        assert_eq!(
            report.failed_cases,
            0,
            "{}",
            serde_json::to_string(&report).unwrap()
        );
        assert!(
            report.passed_cases >= 1,
            "expected at least one ndimage test case"
        );

        let artifacts = write_parity_artifacts(&cfg, &report)
            .expect("ndimage parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn io_packet_runner_passes() {
        let cfg = HarnessConfig::default_paths();
        let report =
            run_io_packet(&cfg, "FSCI-P2C-017_io_core.json").expect("io packet fixture should run");
        assert_eq!(
            report.failed_cases,
            0,
            "{}",
            serde_json::to_string(&report).unwrap()
        );
        assert!(
            report.passed_cases >= 5,
            "expected io fixture coverage across Matrix Market, text, and WAV"
        );

        let artifacts =
            write_parity_artifacts(&cfg, &report).expect("io parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn differential_test_optimize_fixture() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-003_optimize_core.json");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle)
            .expect("differential optimize should succeed");

        assert_eq!(report.packet_id, "FSCI-P2C-003");
        assert_eq!(report.family, "optimize_core");
        assert_eq!(
            report.fail_count,
            0,
            "{}",
            serde_json::to_string_pretty(&report).unwrap()
        );
        assert!(report.pass_count >= 29);
    }

    #[test]
    fn differential_test_interpolate_fixture() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-014_interpolate_core.json");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle)
            .expect("differential interpolate should succeed");

        assert_eq!(report.packet_id, "FSCI-P2C-014");
        assert_eq!(report.family, "interpolate_core");
        assert_eq!(
            report.fail_count,
            0,
            "{}",
            serde_json::to_string_pretty(&report).unwrap()
        );
        assert_eq!(report.pass_count, INTERPOLATE_CORE_CASE_COUNT);
    }

    #[test]
    fn interpolate_cubic_spline_boundary_conditions_present() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-014_interpolate_core.json");
        let fixture: super::InterpolatePacketFixture =
            serde_json::from_slice(&fs::read(&fixture_path).expect("read interpolate fixture"))
                .expect("interpolate fixture should parse");

        let case_ids = fixture
            .cases
            .iter()
            .filter_map(|case| match case {
                super::InterpolateCase::CubicSpline { case_id, .. } => Some(case_id.as_str()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();

        for required in [
            "cubic_spline_natural_nonuniform_grid",
            "cubic_spline_not_a_knot_quadratic_grid",
            "cubic_spline_not_a_knot_wavy_grid",
            "cubic_spline_clamped_zero_slope",
            "cubic_spline_clamped_unit_slopes",
            "cubic_spline_clamped_nan_derivative_hardened_error",
            "cubic_spline_periodic_sine_full_cycle",
            "cubic_spline_periodic_symmetric_wave",
            "cubic_spline_tuple_first_derivative_asymmetric",
            "cubic_spline_tuple_zero_derivative_matches_clamped",
            "cubic_spline_tuple_second_derivative_natural_zero",
        ] {
            assert!(case_ids.contains(required), "missing {required}");
        }
    }

    #[test]
    fn differential_test_ndimage_fixture() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-015_ndimage_core.json");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle)
            .expect("differential ndimage should succeed");

        assert_eq!(report.packet_id, "FSCI-P2C-015");
        assert_eq!(report.family, "ndimage_core");
        assert_eq!(
            report.fail_count,
            0,
            "{}",
            serde_json::to_string_pretty(&report).unwrap()
        );
        assert!(report.pass_count >= 1);
    }

    #[test]
    fn differential_test_io_fixture() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-017_io_core.json");
        let oracle = default_test_oracle();
        let report =
            run_differential_test(&fixture_path, &oracle).expect("differential io should succeed");

        assert_eq!(report.packet_id, "FSCI-P2C-017");
        assert_eq!(report.family, "io_core");
        assert_eq!(
            report.fail_count,
            0,
            "{}",
            serde_json::to_string_pretty(&report).unwrap()
        );
        assert!(report.pass_count >= 5);
    }

    #[test]
    fn interpolate_default_oracle_routes_to_interpolate_script() {
        let path = super::default_differential_oracle_script_path("interpolate_core");
        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some("scipy_interpolate_oracle.py")
        );
    }

    #[test]
    fn io_default_oracle_routes_to_io_script() {
        let path = super::default_differential_oracle_script_path("io_core");
        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some("scipy_io_oracle.py")
        );
    }

    #[test]
    fn ndimage_default_oracle_routes_to_ndimage_script() {
        let path = super::default_differential_oracle_script_path("ndimage_core");
        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some("scipy_ndimage_oracle.py")
        );
    }

    #[test]
    fn differential_test_interpolate_fixture_uses_oracle_capture() {
        let unique = format!("fsci-interpolate-oracle-{}", super::now_unix_ms());
        let root = std::env::temp_dir().join(unique);
        fs::create_dir_all(&root).expect("create temp root");
        let fixture_path = root.join("FSCI-P2C-014_interpolate_oracle.json");
        let fixture = serde_json::json!({
            "packet_id": "FSCI-TEST-P2C-014-MOCK",
            "family": "interpolate_core",
            "cases": [{
                "operation": "interp1d",
                "case_id": "interp1d_oracle_overrides_embedded_expected",
                "category": "differential",
                "mode": "Strict",
                "kind": "linear",
                "x": [0.0, 1.0],
                "y": [0.0, 10.0],
                "x_new": [0.5],
                "expected": {
                    "kind": "vector",
                    "values": [999.0],
                    "atol": 1e-12,
                    "rtol": 1e-12
                }
            }]
        });
        fs::write(
            &fixture_path,
            serde_json::to_vec_pretty(&fixture).expect("serialize interpolate fixture"),
        )
        .expect("write interpolate fixture");

        let script_path = root.join("mock_interpolate_oracle.py");
        let script = r#"
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--fixture", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--oracle-root", required=True)
args = parser.parse_args()

fixture = json.loads(Path(args.fixture).read_text())
result = {
    "packet_id": fixture["packet_id"],
    "family": fixture["family"],
    "generated_unix_ms": 0,
    "runtime": {
        "python_version": "3.11.0",
        "numpy_version": "2.0.0",
        "scipy_version": "mock-1.0",
    },
    "case_outputs": [
        {
            "case_id": "interp1d_oracle_overrides_embedded_expected",
            "status": "ok",
            "result_kind": "vector",
            "result": {"values": [5.0]},
            "error": None,
        }
    ],
}
Path(args.output).write_text(json.dumps(result, indent=2))
"#;
        fs::write(&script_path, script).expect("write mock interpolate oracle");

        let oracle = DifferentialOracleConfig {
            python_path: PathBuf::from("python3"),
            script_path,
            timeout_secs: 30,
            required: true,
        };

        let report = run_differential_test(&fixture_path, &oracle)
            .expect("oracle-backed interpolate fixture runs");

        assert_eq!(report.fail_count, 0);
        assert_eq!(report.pass_count, 1);
        assert_eq!(report.oracle_status, OracleStatus::Available);
        assert!(
            report.per_case_results[0]
                .message
                .contains("interpolate vector matched"),
            "unexpected message: {}",
            report.per_case_results[0].message
        );
    }

    #[test]
    fn differential_test_io_fixture_uses_oracle_capture() {
        let unique = format!("fsci-io-oracle-{}", super::now_unix_ms());
        let root = std::env::temp_dir().join(unique);
        fs::create_dir_all(&root).expect("create temp root");
        let fixture_path = root.join("FSCI-P2C-017_io_oracle.json");
        let fixture = IoPacketFixture {
            packet_id: "FSCI-TEST-P2C-017-MOCK".to_owned(),
            family: "io_core".to_owned(),
            cases: vec![super::IoCase::Mmread {
                case_id: "mmread_oracle_overrides_embedded_expected".to_owned(),
                category: "differential".to_owned(),
                mode: RuntimeMode::Strict,
                content: "%%MatrixMarket matrix array real general\n1 2\n1\n2\n".to_owned(),
                expected: super::IoExpectedOutcome::Matrix {
                    rows: 1,
                    cols: 2,
                    values: vec![999.0, 999.0],
                    atol: Some(1.0e-12),
                    rtol: Some(1.0e-12),
                },
            }],
        };
        fs::write(
            &fixture_path,
            serde_json::to_vec_pretty(&fixture).expect("serialize io fixture"),
        )
        .expect("write io fixture");

        let script_path = root.join("mock_io_oracle.py");
        let script = r#"
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--fixture", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--oracle-root", required=True)
args = parser.parse_args()

fixture = json.loads(Path(args.fixture).read_text())
result = {
    "packet_id": fixture["packet_id"],
    "family": fixture["family"],
    "generated_unix_ms": 0,
    "runtime": {
        "python_version": "3.11.0",
        "numpy_version": "2.0.0",
        "scipy_version": "mock-1.0",
    },
    "case_outputs": [
        {
            "case_id": "mmread_oracle_overrides_embedded_expected",
            "status": "ok",
            "result_kind": "matrix",
            "result": {"rows": 1, "cols": 2, "values": [1.0, 2.0]},
            "error": None,
        }
    ],
}
Path(args.output).write_text(json.dumps(result, indent=2))
"#;
        fs::write(&script_path, script).expect("write mock io oracle");

        let oracle = DifferentialOracleConfig {
            python_path: PathBuf::from("python3"),
            script_path,
            timeout_secs: 30,
            required: true,
        };

        let report =
            run_differential_test(&fixture_path, &oracle).expect("oracle-backed io fixture runs");

        assert_eq!(report.fail_count, 0);
        assert_eq!(report.pass_count, 1);
        assert_eq!(report.oracle_status, OracleStatus::Available);
        assert!(
            report.per_case_results[0]
                .message
                .contains("io matrix matched"),
            "unexpected message: {}",
            report.per_case_results[0].message
        );
    }

    #[test]
    fn differential_optimize_quota_and_structured_logs() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-003_optimize_core.json");
        let raw = fs::read_to_string(&fixture_path).expect("read optimize fixture");
        let fixture: OptimizePacketFixture =
            serde_json::from_str(&raw).expect("parse optimize fixture");
        let oracle = default_test_oracle();
        let report =
            run_differential_test(&fixture_path, &oracle).expect("optimize differential runs");

        let mut by_case = std::collections::BTreeMap::new();
        for case in &fixture.cases {
            by_case.insert(case.case_id().to_owned(), case);
        }

        let mut differential_count = 0usize;
        let mut metamorphic_count = 0usize;
        let mut adversarial_count = 0usize;
        let mut logs = Vec::with_capacity(report.per_case_results.len());

        for case_result in &report.per_case_results {
            let case = by_case
                .get(&case_result.case_id)
                .expect("every report case should exist in fixture");
            match case.category() {
                "differential" => differential_count += 1,
                "metamorphic" => metamorphic_count += 1,
                "adversarial" => adversarial_count += 1,
                _ => {} // Ignore unexpected categories in aggregation
            }

            let log = StructuredCaseLog {
                test_id: case_result.case_id.clone(),
                category: case.category().to_owned(),
                input_summary: optimize_case_input_summary(case),
                expected: optimize_case_expected_summary(case.expected()),
                actual: case_result.message.clone(),
                diff: case_result.max_diff,
                tolerance: case_result.tolerance_used.clone(),
                pass: case_result.passed,
            };

            let payload =
                serde_json::to_string(&log).expect("structured conformance log should serialize");
            let parsed: serde_json::Value =
                serde_json::from_str(&payload).expect("structured log should parse");
            assert!(parsed.get("test_id").is_some());
            assert!(parsed.get("category").is_some());
            assert!(parsed.get("input_summary").is_some());
            assert!(parsed.get("expected").is_some());
            assert!(parsed.get("actual").is_some());
            assert!(parsed.get("diff").is_some());
            assert!(parsed.get("tolerance").is_some());
            assert!(parsed.get("pass").is_some());
            logs.push(log);
        }

        // br-8h5u: derive expected quotas from the fixture so that
        // accidental category drift trips the test. Keep a floor as a
        // sanity check that the fixture itself has not shrunk.
        let (expected_diff, expected_meta, expected_adv) =
            fixture
                .cases
                .iter()
                .fold((0, 0, 0), |(d, m, a), c| match c.category() {
                    "differential" => (d + 1, m, a),
                    "metamorphic" => (d, m + 1, a),
                    "adversarial" => (d, m, a + 1),
                    _ => (d, m, a),
                });
        assert!(
            expected_diff >= 15 && expected_meta >= 6 && expected_adv >= 8,
            "fixture quota floor check: diff={expected_diff}, meta={expected_meta}, adv={expected_adv}"
        );
        assert_eq!(
            differential_count, expected_diff,
            "differential coverage drifted from fixture"
        );
        assert_eq!(
            metamorphic_count, expected_meta,
            "metamorphic coverage drifted from fixture"
        );
        assert_eq!(
            adversarial_count, expected_adv,
            "adversarial coverage drifted from fixture"
        );
        assert_eq!(report.fail_count, 0);

        let output_dir = HarnessConfig::default_paths()
            .artifact_dir_for("P2C-003")
            .join("differential");
        fs::create_dir_all(&output_dir).expect("create optimize differential artifact directory");
        let output_path = output_dir.join("structured_case_logs.json");
        let payload = serde_json::to_vec_pretty(&logs).expect("serialize optimize logs");
        fs::write(&output_path, payload).expect("write optimize structured logs");
        assert!(output_path.exists());
    }

    #[test]
    fn differential_optimize_hardened_minimize_emits_audit_ledger() {
        let unique = format!("fsci-conformance-opt-audit-{}", super::now_unix_ms());
        let root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&root).expect("create temp root");

        let fixture_path = root.join("FSCI-P2C-003_optimize_audit.json");
        let fixture = serde_json::json!({
            "packet_id": "FSCI-P2C-003-AUDIT",
            "family": "optimize_core",
            "cases": [{
                "operation": "minimize",
                "case_id": "hardened_minimize_nan_audit",
                "category": "hardened",
                "mode": "Hardened",
                "method": "Bfgs",
                "objective": "nan_branch",
                "x0": [0.0, 0.0],
                "expected": {
                    "kind": "minimize_status",
                    "status": "NanEncountered",
                    "success": false
                }
            }]
        });
        fs::write(
            &fixture_path,
            serde_json::to_vec_pretty(&fixture).expect("serialize optimize audit fixture"),
        )
        .expect("write optimize audit fixture");

        let report =
            run_differential_test(&fixture_path, &default_test_oracle()).expect("fixture runs");
        assert_eq!(report.fail_count, 0);
        assert_eq!(report.pass_count, 1);

        let audit_path =
            super::differential_audit_ledger_path_for_fixture(&fixture_path, &report.packet_id);
        let audit_jsonl = fs::read_to_string(&audit_path).expect("read optimize audit ledger");
        assert!(audit_path.exists());
        assert!(audit_jsonl.contains("\"kind\":\"fail_closed\""));
        assert!(audit_jsonl.contains("minimize::NanEncountered"));
    }

    #[test]
    fn differential_test_special_fixture() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-006_special_core.json");
        let oracle = default_test_oracle();
        let report =
            run_differential_test(&fixture_path, &oracle).expect("differential special runs");

        assert_eq!(report.packet_id, "FSCI-P2C-006");
        assert_eq!(report.family, "special_core");
        assert_eq!(
            report.fail_count,
            0,
            "{}",
            serde_json::to_string_pretty(&report).unwrap()
        );
        assert!(report.pass_count >= 29);
    }

    #[test]
    fn differential_special_edge_sweep_cases_present() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-006_special_core.json");
        let raw = fs::read_to_string(&fixture_path).expect("read special fixture");
        let fixture: SpecialPacketFixture =
            serde_json::from_str(&raw).expect("parse special fixture");
        let edge_cases: Vec<&SpecialCase> = fixture
            .cases
            .iter()
            .filter(|case| case.case_id().starts_with("edge_"))
            .collect();
        assert!(
            edge_cases.len() >= 16,
            "expected at least 16 edge sweep cases, got {}",
            edge_cases.len()
        );
        assert!(
            edge_cases
                .iter()
                .filter(|case| matches!(case.expected, SpecialExpectedOutcome::Class { .. }))
                .count()
                >= 8,
            "edge sweep should include singularity/class checks"
        );
        assert!(
            edge_cases
                .iter()
                .any(|case| case.function == SpecialCaseFunction::Gamma)
        );
        assert!(
            edge_cases
                .iter()
                .any(|case| case.function == SpecialCaseFunction::Exp1)
        );
        assert!(
            edge_cases
                .iter()
                .any(|case| case.function == SpecialCaseFunction::Gammaln)
        );
        assert!(
            edge_cases
                .iter()
                .any(|case| case.function == SpecialCaseFunction::Digamma)
        );
        assert!(
            edge_cases
                .iter()
                .any(|case| case.function == SpecialCaseFunction::Y0)
        );
        assert!(
            edge_cases
                .iter()
                .any(|case| case.function == SpecialCaseFunction::Kv)
        );
        assert!(
            edge_cases
                .iter()
                .any(|case| case.function == SpecialCaseFunction::Ndtri)
        );
    }

    #[test]
    fn differential_special_elliptic_jacobi_cases_present() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-006_special_core.json");
        let raw = fs::read_to_string(&fixture_path).expect("read special fixture");
        let fixture: SpecialPacketFixture =
            serde_json::from_str(&raw).expect("parse special fixture");
        let elliptic_cases: Vec<&SpecialCase> = fixture
            .cases
            .iter()
            .filter(|case| {
                matches!(
                    case.function,
                    SpecialCaseFunction::Ellipk
                        | SpecialCaseFunction::Ellipkm1
                        | SpecialCaseFunction::Ellipe
                        | SpecialCaseFunction::Ellipkinc
                        | SpecialCaseFunction::Ellipeinc
                        | SpecialCaseFunction::EllipjSn
                        | SpecialCaseFunction::EllipjCn
                        | SpecialCaseFunction::EllipjDn
                        | SpecialCaseFunction::EllipjPh
                )
            })
            .collect();
        assert!(
            elliptic_cases.len() >= 16,
            "expected at least 16 elliptic/Jacobi cases, got {}",
            elliptic_cases.len()
        );
        for function in [
            SpecialCaseFunction::Ellipk,
            SpecialCaseFunction::Ellipkm1,
            SpecialCaseFunction::Ellipe,
            SpecialCaseFunction::Ellipkinc,
            SpecialCaseFunction::Ellipeinc,
            SpecialCaseFunction::EllipjSn,
            SpecialCaseFunction::EllipjCn,
            SpecialCaseFunction::EllipjDn,
            SpecialCaseFunction::EllipjPh,
        ] {
            assert!(
                elliptic_cases.iter().any(|case| case.function == function),
                "missing elliptic/Jacobi fixture coverage for {function:?}"
            );
        }
        assert!(
            elliptic_cases
                .iter()
                .filter(|case| case.mode == RuntimeMode::Hardened)
                .count()
                >= 2,
            "expected hardened elliptic singular-parameter cases"
        );
        assert!(
            elliptic_cases
                .iter()
                .any(|case| matches!(case.expected, SpecialExpectedOutcome::ErrorKind { .. })),
            "expected typed-error coverage for elliptic singular parameters"
        );
    }

    #[test]
    fn differential_special_bessel_gamma_corner_cases_present() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-006_special_core.json");
        let raw = fs::read_to_string(&fixture_path).expect("read special fixture");
        let fixture: SpecialPacketFixture =
            serde_json::from_str(&raw).expect("parse special fixture");
        let corner_cases: Vec<&SpecialCase> = fixture
            .cases
            .iter()
            .filter(|case| case.case_id().starts_with("corner_"))
            .collect();
        assert!(
            corner_cases.len() >= 40,
            "expected at least 40 bessel/gamma corner cases, got {}",
            corner_cases.len()
        );
        for function in [
            SpecialCaseFunction::Gamma,
            SpecialCaseFunction::Gammaln,
            SpecialCaseFunction::Digamma,
            SpecialCaseFunction::Rgamma,
            SpecialCaseFunction::Jn,
            SpecialCaseFunction::Yn,
            SpecialCaseFunction::Iv,
            SpecialCaseFunction::Kv,
            SpecialCaseFunction::SphericalJn,
            SpecialCaseFunction::SphericalYn,
            SpecialCaseFunction::SphericalIn,
            SpecialCaseFunction::SphericalKn,
        ] {
            assert!(
                corner_cases.iter().any(|case| case.function == function),
                "missing corner coverage for {function:?}"
            );
        }
    }

    #[test]
    fn differential_special_quota_and_structured_logs() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-006_special_core.json");
        let raw = fs::read_to_string(&fixture_path).expect("read special fixture");
        let fixture: SpecialPacketFixture =
            serde_json::from_str(&raw).expect("parse special fixture");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle).expect("special differential");

        let mut by_case = std::collections::BTreeMap::new();
        for case in &fixture.cases {
            by_case.insert(case.case_id().to_owned(), case);
        }

        let mut differential_count = 0usize;
        let mut metamorphic_count = 0usize;
        let mut adversarial_count = 0usize;
        let mut logs = Vec::with_capacity(report.per_case_results.len());

        for case_result in &report.per_case_results {
            let case = by_case
                .get(&case_result.case_id)
                .expect("every report case should exist in fixture");
            match case.category() {
                "differential" => differential_count += 1,
                "metamorphic" => metamorphic_count += 1,
                "adversarial" => adversarial_count += 1,
                _ => {} // Ignore unexpected categories in aggregation
            }

            let log = StructuredCaseLog {
                test_id: case_result.case_id.clone(),
                category: case.category().to_owned(),
                input_summary: special_case_input_summary(case),
                expected: special_case_expected_summary(&case.expected),
                actual: case_result.message.clone(),
                diff: case_result.max_diff,
                tolerance: case_result.tolerance_used.clone(),
                pass: case_result.passed,
            };

            let payload =
                serde_json::to_string(&log).expect("structured conformance log should serialize");
            let parsed: serde_json::Value =
                serde_json::from_str(&payload).expect("structured log should parse");
            assert!(parsed.get("test_id").is_some());
            assert!(parsed.get("category").is_some());
            assert!(parsed.get("input_summary").is_some());
            assert!(parsed.get("expected").is_some());
            assert!(parsed.get("actual").is_some());
            assert!(parsed.get("diff").is_some());
            assert!(parsed.get("tolerance").is_some());
            assert!(parsed.get("pass").is_some());
            logs.push(log);
        }

        // br-8h5u: derive expected quotas from the fixture; floor below
        // guards against fixture shrinkage.
        let (expected_diff, expected_meta, expected_adv) =
            fixture
                .cases
                .iter()
                .fold((0, 0, 0), |(d, m, a), c| match c.category() {
                    "differential" => (d + 1, m, a),
                    "metamorphic" => (d, m + 1, a),
                    "adversarial" => (d, m, a + 1),
                    _ => (d, m, a),
                });
        assert!(
            expected_diff >= 15 && expected_meta >= 6 && expected_adv >= 8,
            "fixture quota floor check: diff={expected_diff}, meta={expected_meta}, adv={expected_adv}"
        );
        assert_eq!(
            differential_count, expected_diff,
            "differential coverage drifted from fixture"
        );
        assert_eq!(
            metamorphic_count, expected_meta,
            "metamorphic coverage drifted from fixture"
        );
        assert_eq!(
            adversarial_count, expected_adv,
            "adversarial coverage drifted from fixture"
        );
        assert_eq!(report.fail_count, 0);

        let output_dir = HarnessConfig::default_paths()
            .artifact_dir_for("P2C-006")
            .join("differential");
        fs::create_dir_all(&output_dir).expect("create special differential artifact directory");
        let output_path = output_dir.join("structured_case_logs.json");
        let payload = serde_json::to_vec_pretty(&logs).expect("serialize special logs");
        fs::write(&output_path, payload).expect("write special structured logs");
        assert!(output_path.exists());
    }

    #[test]
    fn differential_test_array_api_fixture() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-007_arrayapi_core.json");
        let oracle = default_test_oracle();
        let report =
            run_differential_test(&fixture_path, &oracle).expect("differential array_api runs");

        assert_eq!(report.packet_id, "FSCI-P2C-007");
        assert_eq!(report.family, "array_api_core");
        assert_eq!(
            report.fail_count,
            0,
            "{}",
            serde_json::to_string_pretty(&report).unwrap()
        );
        assert!(report.pass_count >= 29);
    }

    #[test]
    fn array_api_contract_table_matches_fixture_contract_refs() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-007_arrayapi_core.json");
        let raw = fs::read_to_string(&fixture_path).expect("read array api fixture");
        let fixture: ArrayApiPacketFixture =
            serde_json::from_str(&raw).expect("parse array api fixture");
        let table = load_array_api_contract_table().expect("load array api contract table");

        assert_eq!(table.packet_id.as_deref(), Some("FSCI-P2C-007"));
        assert_eq!(table.domain.as_deref(), Some("arrayapi"));

        let fixture_refs = fixture
            .cases
            .iter()
            .filter_map(|case| match case.expected() {
                ArrayApiExpectedOutcome::Array { contract_ref, .. }
                | ArrayApiExpectedOutcome::Shape { contract_ref, .. }
                | ArrayApiExpectedOutcome::Dtype { contract_ref, .. }
                | ArrayApiExpectedOutcome::Bool { contract_ref, .. } => contract_ref.clone(),
                ArrayApiExpectedOutcome::ErrorKind { .. } => None,
            })
            .collect::<std::collections::BTreeSet<_>>();
        let table_refs = table
            .contracts
            .iter()
            .map(|entry| entry.function_name.clone())
            .collect::<std::collections::BTreeSet<_>>();

        assert_eq!(fixture_refs.len(), 9);
        assert_eq!(table_refs, fixture_refs);
    }

    fn collect_contract_refs(value: &serde_json::Value, refs: &mut BTreeSet<String>) {
        match value {
            serde_json::Value::Object(map) => {
                if let Some(serde_json::Value::String(contract_ref)) = map.get("contract_ref")
                    && !contract_ref.is_empty()
                {
                    refs.insert(contract_ref.clone());
                }

                for child in map.values() {
                    collect_contract_refs(child, refs);
                }
            }
            serde_json::Value::Array(values) => {
                for child in values {
                    collect_contract_refs(child, refs);
                }
            }
            _ => {}
        }
    }

    fn contract_table_function_names(packet_id: &str) -> BTreeSet<String> {
        let table_path = HarnessConfig::default_paths()
            .artifact_dir_for(packet_id)
            .join("contracts/contract_table.json");
        let raw = fs::read_to_string(&table_path).expect("read contract table");
        let table: serde_json::Value = serde_json::from_str(&raw).expect("parse contract table");
        table
            .get("contracts")
            .and_then(serde_json::Value::as_array)
            .expect("contract table has contracts array")
            .iter()
            .map(|entry| {
                entry
                    .get("function_name")
                    .and_then(serde_json::Value::as_str)
                    .expect("contract row has function_name")
                    .to_owned()
            })
            .collect()
    }

    #[test]
    fn artifact_contract_tables_cover_fixture_contract_refs() {
        let cfg = HarnessConfig::default_paths();
        for (packet_id, fixture_name) in [
            ("P2C-003", "FSCI-P2C-003_optimize_core.json"),
            ("P2C-006", "FSCI-P2C-006_special_core.json"),
            ("P2C-007", "FSCI-P2C-007_arrayapi_core.json"),
            ("P2C-009", "FSCI-P2C-009_cluster_core.json"),
            ("P2C-010", "FSCI-P2C-010_spatial_core.json"),
            ("P2C-011", "FSCI-P2C-011_signal_core.json"),
            ("P2C-012", "FSCI-P2C-012_stats_core.json"),
            ("P2C-013", "FSCI-P2C-013_integrate_core.json"),
        ] {
            let fixture_path = cfg.fixture_root.join(fixture_name);
            let raw = fs::read_to_string(&fixture_path).expect("read fixture");
            let fixture: serde_json::Value = serde_json::from_str(&raw).expect("parse fixture");
            let mut fixture_refs = BTreeSet::new();
            collect_contract_refs(&fixture, &mut fixture_refs);

            let table_refs = contract_table_function_names(packet_id);
            let missing = fixture_refs
                .difference(&table_refs)
                .cloned()
                .collect::<Vec<_>>();
            let extra = table_refs
                .difference(&fixture_refs)
                .cloned()
                .collect::<Vec<_>>();

            assert!(
                missing.is_empty(),
                "{packet_id} contract table missing fixture refs: {}",
                missing.join(", ")
            );
            assert!(
                extra.is_empty(),
                "{packet_id} contract table has rows without fixture refs: {}",
                extra.join(", ")
            );
        }
    }

    #[test]
    fn expanded_contract_table_loaders_provide_family_tolerances() {
        for tolerance in [
            super::resolve_cluster_contract_tolerance(Some("fsci_cluster::kmeans"), None, None),
            super::resolve_spatial_contract_tolerance(Some("fsci_spatial::pdist"), None, None),
            super::resolve_signal_contract_tolerance(Some("fsci_signal::hann"), None, None),
            super::resolve_stats_contract_tolerance(Some("fsci_stats::describe"), None, None),
            super::resolve_integrate_contract_tolerance(
                Some("fsci_integrate::solve_ivp"),
                None,
                None,
            ),
        ] {
            assert_eq!(tolerance.comparison_mode, "allclose");
            assert_eq!(tolerance.atol, 1.0e-10);
            assert_eq!(tolerance.rtol, 1.0e-8);
        }
    }

    #[test]
    fn array_api_contract_table_provides_expected_tolerances() {
        let linspace = resolve_array_api_contract_tolerance(
            Some("fsci_arrayapi::creation::linspace"),
            None,
            None,
        );
        assert_eq!(linspace.comparison_mode, "mixed");
        assert_eq!(linspace.atol, 1.0e-12);
        assert_eq!(linspace.rtol, 1.0e-12);

        let zeros = resolve_array_api_contract_tolerance(
            Some("fsci_arrayapi::creation::zeros"),
            None,
            None,
        );
        assert_eq!(zeros.comparison_mode, "exact");
        assert_eq!(zeros.atol, 0.0);
        assert_eq!(zeros.rtol, 0.0);
    }

    #[test]
    fn differential_array_api_quota_and_structured_logs() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-007_arrayapi_core.json");
        let raw = fs::read_to_string(&fixture_path).expect("read array api fixture");
        let fixture: ArrayApiPacketFixture =
            serde_json::from_str(&raw).expect("parse array api fixture");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle).expect("array api differential");

        let mut by_case = std::collections::BTreeMap::new();
        for case in &fixture.cases {
            by_case.insert(case.case_id().to_owned(), case);
        }

        let mut differential_count = 0usize;
        let mut metamorphic_count = 0usize;
        let mut adversarial_count = 0usize;
        let mut logs = Vec::with_capacity(report.per_case_results.len());

        for case_result in &report.per_case_results {
            let case = by_case
                .get(&case_result.case_id)
                .expect("every report case should exist in fixture");
            match case.category() {
                "differential" => differential_count += 1,
                "metamorphic" => metamorphic_count += 1,
                "adversarial" => adversarial_count += 1,
                _ => {} // Ignore unexpected categories in aggregation
            }

            let log = StructuredCaseLog {
                test_id: case_result.case_id.clone(),
                category: case.category().to_owned(),
                input_summary: array_api_case_input_summary(case),
                expected: array_api_case_expected_summary(case.expected()),
                actual: case_result.message.clone(),
                diff: case_result.max_diff,
                tolerance: case_result.tolerance_used.clone(),
                pass: case_result.passed,
            };

            let payload =
                serde_json::to_string(&log).expect("structured conformance log should serialize");
            let parsed: serde_json::Value =
                serde_json::from_str(&payload).expect("structured log should parse");
            assert!(parsed.get("test_id").is_some());
            assert!(parsed.get("category").is_some());
            assert!(parsed.get("input_summary").is_some());
            assert!(parsed.get("expected").is_some());
            assert!(parsed.get("actual").is_some());
            assert!(parsed.get("diff").is_some());
            assert!(parsed.get("tolerance").is_some());
            assert!(parsed.get("pass").is_some());
            logs.push(log);
        }

        // br-8h5u: derive quotas from fixture; floors guard against shrinkage.
        let (expected_diff, expected_meta, expected_adv) =
            fixture
                .cases
                .iter()
                .fold((0, 0, 0), |(d, m, a), c| match c.category() {
                    "differential" => (d + 1, m, a),
                    "metamorphic" => (d, m + 1, a),
                    "adversarial" => (d, m, a + 1),
                    _ => (d, m, a),
                });
        assert!(
            expected_diff >= 15 && expected_meta >= 6 && expected_adv >= 8,
            "fixture quota floor check: diff={expected_diff}, meta={expected_meta}, adv={expected_adv}"
        );
        assert_eq!(
            differential_count, expected_diff,
            "differential coverage drifted from fixture"
        );
        assert_eq!(
            metamorphic_count, expected_meta,
            "metamorphic coverage drifted from fixture"
        );
        assert_eq!(
            adversarial_count, expected_adv,
            "adversarial coverage drifted from fixture"
        );
        assert_eq!(report.fail_count, 0);

        let output_dir = HarnessConfig::default_paths()
            .artifact_dir_for("P2C-007")
            .join("differential");
        fs::create_dir_all(&output_dir).expect("create array_api differential artifact directory");
        let output_path = output_dir.join("structured_case_logs.json");
        let payload = serde_json::to_vec_pretty(&logs).expect("serialize array api logs");
        fs::write(&output_path, payload).expect("write array_api structured logs");
        assert!(output_path.exists());
    }

    #[test]
    fn differential_array_api_hardened_rejections_emit_audit_ledger() {
        let unique = format!("fsci-conformance-array-api-audit-{}", super::now_unix_ms());
        let root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&root).expect("create temp root");

        let fixture_path = root.join("FSCI-P2C-007_arrayapi_audit.json");
        let fixture = serde_json::json!({
            "packet_id": "FSCI-P2C-007-AUDIT",
            "family": "array_api_core",
            "cases": [
                {
                    "operation": "arange",
                    "case_id": "hardened_arange_zero_step_audit",
                    "category": "hardened",
                    "mode": "Hardened",
                    "start": { "kind": "i64", "value": 0 },
                    "stop": { "kind": "i64", "value": 4 },
                    "step": { "kind": "i64", "value": 0 },
                    "dtype": "float64",
                    "expected": {
                        "kind": "error_kind",
                        "error": "InvalidStep"
                    }
                },
                {
                    "operation": "from_slice",
                    "case_id": "hardened_from_slice_mismatch_audit",
                    "category": "hardened",
                    "mode": "Hardened",
                    "values": [
                        { "kind": "f64", "value": 1.0 },
                        { "kind": "f64", "value": 2.0 },
                        { "kind": "f64", "value": 3.0 }
                    ],
                    "shape": [2, 2],
                    "dtype": "float64",
                    "expected": {
                        "kind": "error_kind",
                        "error": "InvalidShape"
                    }
                },
                {
                    "operation": "getitem",
                    "case_id": "hardened_getitem_mode_mismatch_audit",
                    "category": "hardened",
                    "mode": "Hardened",
                    "source_values": [
                        { "kind": "f64", "value": 1.0 },
                        { "kind": "f64", "value": 2.0 },
                        { "kind": "f64", "value": 3.0 }
                    ],
                    "source_shape": [3],
                    "source_dtype": "float64",
                    "indexing_mode": "basic",
                    "index": {
                        "kind": "advanced",
                        "indices": [[0]]
                    },
                    "expected": {
                        "kind": "error_kind",
                        "error": "InvalidIndex"
                    }
                },
                {
                    "operation": "reshape",
                    "case_id": "hardened_reshape_size_mismatch_audit",
                    "category": "hardened",
                    "mode": "Hardened",
                    "source_values": [
                        { "kind": "f64", "value": 1.0 },
                        { "kind": "f64", "value": 2.0 },
                        { "kind": "f64", "value": 3.0 },
                        { "kind": "f64", "value": 4.0 }
                    ],
                    "source_shape": [4],
                    "source_dtype": "float64",
                    "new_shape": [3, 2],
                    "expected": {
                        "kind": "error_kind",
                        "error": "InvalidShape"
                    }
                },
                {
                    "operation": "broadcast_shapes",
                    "case_id": "hardened_broadcast_incompatible_audit",
                    "category": "hardened",
                    "mode": "Hardened",
                    "shapes": [[2, 3], [3, 2]],
                    "expected": {
                        "kind": "error_kind",
                        "error": "BroadcastIncompatible"
                    }
                }
            ]
        });
        fs::write(
            &fixture_path,
            serde_json::to_vec_pretty(&fixture).expect("serialize array api audit fixture"),
        )
        .expect("write array api audit fixture");

        let report =
            run_differential_test(&fixture_path, &default_test_oracle()).expect("fixture runs");
        assert_eq!(report.fail_count, 0);
        assert_eq!(report.pass_count, 5);

        let audit_path =
            super::differential_audit_ledger_path_for_fixture(&fixture_path, &report.packet_id);
        let audit_jsonl = fs::read_to_string(&audit_path).expect("read array api audit ledger");
        assert!(audit_path.exists());
        assert_eq!(audit_jsonl.lines().count(), 5);
        assert!(audit_jsonl.contains("\"kind\":\"fail_closed\""));
        assert!(audit_jsonl.contains("arange::InvalidStep"));
        assert!(audit_jsonl.contains("from_slice::InvalidShape"));
        assert!(audit_jsonl.contains("getitem::InvalidIndex"));
        assert!(audit_jsonl.contains("reshape::InvalidShape"));
        assert!(audit_jsonl.contains("broadcast_shapes::BroadcastIncompatible"));
    }

    #[test]
    fn differential_test_integrate_fixture() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-013_integrate_core.json");
        let oracle = default_test_oracle();
        let report =
            run_differential_test(&fixture_path, &oracle).expect("differential integrate runs");

        assert_eq!(report.packet_id, "FSCI-P2C-013");
        assert_eq!(report.family, "integrate_core");
        assert_eq!(
            report.fail_count,
            0,
            "{}",
            serde_json::to_string_pretty(&report).unwrap()
        );
        assert!(report.pass_count >= 23);
    }

    #[test]
    fn differential_integrate_quota_and_structured_logs() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-013_integrate_core.json");
        let raw = fs::read_to_string(&fixture_path).expect("read integrate fixture");
        let fixture: IntegratePacketFixture =
            serde_json::from_str(&raw).expect("parse integrate fixture");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle).expect("integrate differential");

        let mut by_case = std::collections::BTreeMap::new();
        for case in &fixture.cases {
            by_case.insert(case.case_id().to_owned(), case);
        }

        let mut differential_count = 0usize;
        let mut logs = Vec::with_capacity(report.per_case_results.len());

        for case_result in &report.per_case_results {
            let case = by_case
                .get(&case_result.case_id)
                .expect("every report case should exist in fixture");
            if case.category == "differential" {
                differential_count += 1;
            }

            let log = StructuredCaseLog {
                test_id: case_result.case_id.clone(),
                category: case.category.clone(),
                input_summary: integrate_case_input_summary(case),
                expected: integrate_case_expected_summary(&case.expected),
                actual: case_result.message.clone(),
                diff: case_result.max_diff,
                tolerance: case_result.tolerance_used.clone(),
                pass: case_result.passed,
            };

            let payload =
                serde_json::to_string(&log).expect("structured conformance log should serialize");
            let parsed: serde_json::Value =
                serde_json::from_str(&payload).expect("structured log should parse");
            assert!(parsed.get("test_id").is_some());
            assert!(parsed.get("category").is_some());
            assert!(parsed.get("input_summary").is_some());
            assert!(parsed.get("expected").is_some());
            assert!(parsed.get("actual").is_some());
            assert!(parsed.get("diff").is_some());
            assert!(parsed.get("tolerance").is_some());
            assert!(parsed.get("pass").is_some());
            logs.push(log);
        }

        assert!(
            differential_count >= 23,
            "expected >=23 differential cases, got {differential_count}"
        );
        assert_eq!(report.fail_count, 0);

        let output_dir = HarnessConfig::default_paths()
            .artifact_dir_for("P2C-013")
            .join("differential");
        fs::create_dir_all(&output_dir).expect("create integrate differential artifact directory");
        let output_path = output_dir.join("structured_case_logs.json");
        let payload = serde_json::to_vec_pretty(&logs).expect("serialize integrate logs");
        fs::write(&output_path, payload).expect("write integrate structured logs");
        assert!(output_path.exists());
    }

    #[test]
    fn differential_test_stats_fixture() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-012_stats_core.json");
        let oracle = default_test_oracle();
        let report =
            run_differential_test(&fixture_path, &oracle).expect("differential stats runs");

        assert_eq!(report.packet_id, "FSCI-P2C-012");
        assert_eq!(report.family, "stats_core");
        assert_eq!(
            report.fail_count,
            0,
            "{}",
            serde_json::to_string_pretty(&report).unwrap()
        );
        assert!(report.pass_count >= 30);
    }

    #[test]
    fn differential_stats_quota_and_structured_logs() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-012_stats_core.json");
        let raw = fs::read_to_string(&fixture_path).expect("read stats fixture");
        let fixture: StatsPacketFixture = serde_json::from_str(&raw).expect("parse stats fixture");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle).expect("stats differential");

        let mut by_case = std::collections::BTreeMap::new();
        for case in &fixture.cases {
            by_case.insert(case.case_id().to_owned(), case);
        }

        let mut differential_count = 0usize;
        let mut logs = Vec::with_capacity(report.per_case_results.len());

        for case_result in &report.per_case_results {
            let case = by_case
                .get(&case_result.case_id)
                .expect("every report case should exist in fixture");
            if case.category == "differential" {
                differential_count += 1;
            }

            let log = StructuredCaseLog {
                test_id: case_result.case_id.clone(),
                category: case.category.clone(),
                input_summary: stats_case_input_summary(case),
                expected: stats_case_expected_summary(&case.expected),
                actual: case_result.message.clone(),
                diff: case_result.max_diff,
                tolerance: case_result.tolerance_used.clone(),
                pass: case_result.passed,
            };

            let payload =
                serde_json::to_string(&log).expect("structured conformance log should serialize");
            let parsed: serde_json::Value =
                serde_json::from_str(&payload).expect("structured log should parse");
            assert!(parsed.get("test_id").is_some());
            assert!(parsed.get("category").is_some());
            assert!(parsed.get("input_summary").is_some());
            assert!(parsed.get("expected").is_some());
            assert!(parsed.get("actual").is_some());
            assert!(parsed.get("diff").is_some());
            assert!(parsed.get("tolerance").is_some());
            assert!(parsed.get("pass").is_some());
            logs.push(log);
        }

        assert!(
            differential_count >= 30,
            "expected >=30 differential cases, got {differential_count}"
        );
        assert_eq!(report.fail_count, 0);

        let output_dir = HarnessConfig::default_paths()
            .artifact_dir_for("P2C-012")
            .join("differential");
        fs::create_dir_all(&output_dir).expect("create stats differential artifact directory");
        let output_path = output_dir.join("structured_case_logs.json");
        let payload = serde_json::to_vec_pretty(&logs).expect("serialize stats logs");
        fs::write(&output_path, payload).expect("write stats structured logs");
        assert!(output_path.exists());
    }

    #[test]
    fn differential_test_signal_fixture() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-011_signal_core.json");
        let oracle = default_test_oracle();
        let report =
            run_differential_test(&fixture_path, &oracle).expect("differential signal runs");

        assert_eq!(report.packet_id, "FSCI-P2C-011");
        assert_eq!(report.family, "signal_core");
        assert_eq!(
            report.fail_count,
            0,
            "{}",
            serde_json::to_string_pretty(&report).unwrap()
        );
        assert!(report.pass_count >= 12);
    }

    #[test]
    fn differential_signal_iir_matrix_covers_all_design_families() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-011_signal_core.json");
        let raw = fs::read_to_string(&fixture_path).expect("read signal fixture");
        let fixture: SignalPacketFixture =
            serde_json::from_str(&raw).expect("parse signal fixture");

        let expected_btypes: BTreeSet<String> = ["low", "high", "bandpass", "bandstop"]
            .iter()
            .map(|value| (*value).to_owned())
            .collect();
        let mut by_family = std::collections::BTreeMap::<String, BTreeSet<String>>::new();
        let mut hardened_error_families = BTreeSet::new();
        let mut coefficient_case_count = 0usize;

        for case in &fixture.cases {
            if !matches!(
                case.function.as_str(),
                "butter" | "cheby1" | "cheby2" | "ellip" | "bessel"
            ) {
                continue;
            }

            match case.expected.kind.as_str() {
                "iir_coeffs" => {
                    coefficient_case_count += 1;
                    let btype = case
                        .args
                        .last()
                        .and_then(serde_json::Value::as_str)
                        .expect("IIR coefficient case has btype as last argument");
                    by_family
                        .entry(case.function.clone())
                        .or_default()
                        .insert(btype.to_owned());
                    assert!(
                        case.expected.b.as_ref().is_some_and(|b| !b.is_empty()),
                        "{} should freeze numerator coefficients",
                        case.case_id
                    );
                    assert!(
                        case.expected.a.as_ref().is_some_and(|a| !a.is_empty()),
                        "{} should freeze denominator coefficients",
                        case.case_id
                    );
                }
                "error" if case.case_id.starts_with("hardened_iir_") => {
                    hardened_error_families.insert(case.function.clone());
                }
                _ => {}
            }
        }

        assert!(
            coefficient_case_count >= 20,
            "expected >=20 IIR coefficient cases, got {coefficient_case_count}"
        );
        for family in ["butter", "cheby1", "cheby2", "ellip", "bessel"] {
            assert_eq!(
                by_family.get(family),
                Some(&expected_btypes),
                "{family} should cover low/high/bandpass/bandstop coefficient parity"
            );
            assert!(
                hardened_error_families.contains(family),
                "{family} should have a hardened invalid-Wn case"
            );
        }
    }

    #[test]
    fn differential_signal_quota_and_structured_logs() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-011_signal_core.json");
        let raw = fs::read_to_string(&fixture_path).expect("read signal fixture");
        let fixture: SignalPacketFixture =
            serde_json::from_str(&raw).expect("parse signal fixture");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle).expect("signal differential");

        let mut by_case = std::collections::BTreeMap::new();
        for case in &fixture.cases {
            by_case.insert(case.case_id().to_owned(), case);
        }

        let mut differential_count = 0usize;
        let mut logs = Vec::with_capacity(report.per_case_results.len());

        for case_result in &report.per_case_results {
            let case = by_case
                .get(&case_result.case_id)
                .expect("every report case should exist in fixture");
            if case.category == "differential" {
                differential_count += 1;
            }

            let log = StructuredCaseLog {
                test_id: case_result.case_id.clone(),
                category: case.category.clone(),
                input_summary: signal_case_input_summary(case),
                expected: signal_case_expected_summary(&case.expected),
                actual: case_result.message.clone(),
                diff: case_result.max_diff,
                tolerance: case_result.tolerance_used.clone(),
                pass: case_result.passed,
            };

            let payload =
                serde_json::to_string(&log).expect("structured conformance log should serialize");
            let parsed: serde_json::Value =
                serde_json::from_str(&payload).expect("structured log should parse");
            assert!(parsed.get("test_id").is_some());
            assert!(parsed.get("category").is_some());
            assert!(parsed.get("input_summary").is_some());
            assert!(parsed.get("expected").is_some());
            assert!(parsed.get("actual").is_some());
            assert!(parsed.get("diff").is_some());
            assert!(parsed.get("tolerance").is_some());
            assert!(parsed.get("pass").is_some());
            logs.push(log);
        }

        assert!(
            differential_count >= 12,
            "expected >=12 differential cases, got {differential_count}"
        );
        assert_eq!(report.fail_count, 0);

        let output_dir = HarnessConfig::default_paths()
            .artifact_dir_for("P2C-011")
            .join("differential");
        fs::create_dir_all(&output_dir).expect("create signal differential artifact directory");
        let output_path = output_dir.join("structured_case_logs.json");
        let payload = serde_json::to_vec_pretty(&logs).expect("serialize signal logs");
        fs::write(&output_path, payload).expect("write signal structured logs");
        assert!(output_path.exists());
    }

    #[test]
    fn differential_test_spatial_fixture() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-010_spatial_core.json");
        let oracle = default_test_oracle();
        let report =
            run_differential_test(&fixture_path, &oracle).expect("differential spatial runs");

        assert_eq!(report.packet_id, "FSCI-P2C-010");
        assert_eq!(report.family, "spatial_core");
        assert_eq!(
            report.fail_count,
            0,
            "{}",
            serde_json::to_string_pretty(&report).unwrap()
        );
        assert!(report.pass_count >= 20);
    }

    #[test]
    fn differential_spatial_quota_and_structured_logs() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-010_spatial_core.json");
        let raw = fs::read_to_string(&fixture_path).expect("read spatial fixture");
        let fixture: SpatialPacketFixture =
            serde_json::from_str(&raw).expect("parse spatial fixture");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle).expect("spatial differential");

        let mut by_case = std::collections::BTreeMap::new();
        for case in &fixture.cases {
            by_case.insert(case.case_id().to_owned(), case);
        }

        let mut differential_count = 0usize;
        let mut logs = Vec::with_capacity(report.per_case_results.len());

        for case_result in &report.per_case_results {
            let case = by_case
                .get(&case_result.case_id)
                .expect("every report case should exist in fixture");
            if case.category == "differential" {
                differential_count += 1;
            }

            let log = StructuredCaseLog {
                test_id: case_result.case_id.clone(),
                category: case.category.clone(),
                input_summary: spatial_case_input_summary(case),
                expected: spatial_case_expected_summary(&case.expected),
                actual: case_result.message.clone(),
                diff: case_result.max_diff,
                tolerance: case_result.tolerance_used.clone(),
                pass: case_result.passed,
            };

            let payload =
                serde_json::to_string(&log).expect("structured conformance log should serialize");
            let parsed: serde_json::Value =
                serde_json::from_str(&payload).expect("structured log should parse");
            assert!(parsed.get("test_id").is_some());
            assert!(parsed.get("category").is_some());
            assert!(parsed.get("input_summary").is_some());
            assert!(parsed.get("expected").is_some());
            assert!(parsed.get("actual").is_some());
            assert!(parsed.get("diff").is_some());
            assert!(parsed.get("tolerance").is_some());
            assert!(parsed.get("pass").is_some());
            logs.push(log);
        }

        assert!(
            differential_count >= 20,
            "expected >=20 differential cases, got {differential_count}"
        );
        assert_eq!(report.fail_count, 0);

        let output_dir = HarnessConfig::default_paths()
            .artifact_dir_for("P2C-010")
            .join("differential");
        fs::create_dir_all(&output_dir).expect("create spatial differential artifact directory");
        let output_path = output_dir.join("structured_case_logs.json");
        let payload = serde_json::to_vec_pretty(&logs).expect("serialize spatial logs");
        fs::write(&output_path, payload).expect("write spatial structured logs");
        assert!(output_path.exists());
    }

    #[test]
    fn differential_test_constants_fixture() {
        // frankenscipy-utus: FSCI-P2C-016 was a dead fixture (prose-only
        // schema, no dispatch arm). After rewrite it must parse, dispatch
        // through run_differential_constants, and match fsci-constants
        // values bit-exactly against fixture expected values.
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-016_constants_core.json");
        let oracle = default_test_oracle();
        let report =
            run_differential_test(&fixture_path, &oracle).expect("differential constants runs");

        assert_eq!(report.packet_id, "FSCI-P2C-016");
        assert_eq!(report.family, "constants_core");
        assert_eq!(
            report.fail_count,
            0,
            "{}",
            serde_json::to_string_pretty(&report).unwrap()
        );
        assert!(report.pass_count >= 20);
    }

    #[test]
    fn differential_test_cluster_fixture() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-009_cluster_core.json");
        let oracle = default_test_oracle();
        let report =
            run_differential_test(&fixture_path, &oracle).expect("differential cluster runs");

        assert_eq!(report.packet_id, "FSCI-P2C-009");
        assert_eq!(report.family, "cluster_core");
        assert_eq!(
            report.fail_count,
            0,
            "{}",
            serde_json::to_string_pretty(&report).unwrap()
        );
        assert!(report.pass_count >= 20);
    }

    #[test]
    fn differential_cluster_quota_and_structured_logs() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-009_cluster_core.json");
        let raw = fs::read_to_string(&fixture_path).expect("read cluster fixture");
        let fixture: ClusterPacketFixture =
            serde_json::from_str(&raw).expect("parse cluster fixture");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle).expect("cluster differential");

        let mut by_case = std::collections::BTreeMap::new();
        for case in &fixture.cases {
            by_case.insert(case.case_id().to_owned(), case);
        }

        let mut differential_count = 0usize;
        let mut logs = Vec::with_capacity(report.per_case_results.len());

        for case_result in &report.per_case_results {
            let case = by_case
                .get(&case_result.case_id)
                .expect("every report case should exist in fixture");
            if case.category == "differential" {
                differential_count += 1;
            }

            let log = StructuredCaseLog {
                test_id: case_result.case_id.clone(),
                category: case.category.clone(),
                input_summary: cluster_case_input_summary(case),
                expected: cluster_case_expected_summary(&case.expected),
                actual: case_result.message.clone(),
                diff: case_result.max_diff,
                tolerance: case_result.tolerance_used.clone(),
                pass: case_result.passed,
            };

            let payload =
                serde_json::to_string(&log).expect("structured conformance log should serialize");
            let parsed: serde_json::Value =
                serde_json::from_str(&payload).expect("structured log should parse");
            assert!(parsed.get("test_id").is_some());
            assert!(parsed.get("category").is_some());
            assert!(parsed.get("input_summary").is_some());
            assert!(parsed.get("expected").is_some());
            assert!(parsed.get("actual").is_some());
            assert!(parsed.get("diff").is_some());
            assert!(parsed.get("tolerance").is_some());
            assert!(parsed.get("pass").is_some());
            logs.push(log);
        }

        assert!(
            differential_count >= 20,
            "expected >=20 differential cases, got {differential_count}"
        );
        assert_eq!(report.fail_count, 0);

        let output_dir = HarnessConfig::default_paths()
            .artifact_dir_for("P2C-009")
            .join("differential");
        fs::create_dir_all(&output_dir).expect("create cluster differential artifact directory");
        let output_path = output_dir.join("structured_case_logs.json");
        let payload = serde_json::to_vec_pretty(&logs).expect("serialize cluster logs");
        fs::write(&output_path, payload).expect("write cluster structured logs");
        assert!(output_path.exists());
    }

    #[test]
    fn differential_test_casp_fixture() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-008_runtime_casp.json");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle).expect("differential casp runs");

        assert_eq!(report.packet_id, "FSCI-P2C-008");
        assert_eq!(report.family, "runtime_casp");
        assert_eq!(
            report.fail_count,
            0,
            "{}",
            serde_json::to_string_pretty(&report).unwrap()
        );
        assert!(report.pass_count >= 15);
    }

    #[test]
    fn differential_casp_quota_and_structured_logs() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-008_runtime_casp.json");
        let raw = fs::read_to_string(&fixture_path).expect("read casp fixture");
        let fixture: CaspPacketFixture = serde_json::from_str(&raw).expect("parse casp fixture");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle).expect("casp differential");

        let mut by_case = std::collections::BTreeMap::new();
        for case in &fixture.cases {
            by_case.insert(case.case_id().to_owned(), case);
        }

        let mut case_count = 0usize;
        let mut logs = Vec::with_capacity(report.per_case_results.len());

        for case_result in &report.per_case_results {
            let case = by_case
                .get(&case_result.case_id)
                .expect("every report case should exist in fixture");
            case_count += 1;

            let log = StructuredCaseLog {
                test_id: case_result.case_id.clone(),
                category: case.category.clone(),
                input_summary: casp_case_input_summary(case),
                expected: casp_case_expected_summary(&case.expected),
                actual: case_result.message.clone(),
                diff: case_result.max_diff,
                tolerance: case_result.tolerance_used.clone(),
                pass: case_result.passed,
            };

            let payload =
                serde_json::to_string(&log).expect("structured conformance log should serialize");
            let parsed: serde_json::Value =
                serde_json::from_str(&payload).expect("structured log should parse");
            assert!(parsed.get("test_id").is_some());
            assert!(parsed.get("category").is_some());
            assert!(parsed.get("input_summary").is_some());
            assert!(parsed.get("expected").is_some());
            assert!(parsed.get("actual").is_some());
            assert!(parsed.get("diff").is_some());
            assert!(parsed.get("tolerance").is_some());
            assert!(parsed.get("pass").is_some());
            logs.push(log);
        }

        assert_eq!(case_count, 15, "expected 15 cases, got {case_count}");
        assert_eq!(report.fail_count, 0);

        let output_dir = HarnessConfig::default_paths()
            .artifact_dir_for("P2C-008")
            .join("differential");
        fs::create_dir_all(&output_dir).expect("create casp differential artifact directory");
        let output_path = output_dir.join("structured_case_logs.json");
        let payload = serde_json::to_vec_pretty(&logs).expect("serialize casp logs");
        fs::write(&output_path, payload).expect("write casp structured logs");
        assert!(output_path.exists());
    }

    #[test]
    fn differential_test_fft_fixture() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-005_fft_core.json");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle).expect("differential fft runs");

        assert_eq!(report.packet_id, "FSCI-P2C-005");
        assert_eq!(report.family, "fft_core");
        assert_eq!(
            report.fail_count,
            0,
            "{}",
            serde_json::to_string_pretty(&report).unwrap()
        );
        assert!(report.pass_count >= 56);
    }

    #[test]
    fn linalg_oracle_error_and_rust_error_is_not_an_auto_pass() {
        let case = LinalgCase::Solve {
            case_id: "oracle_missing_solve".to_owned(),
            mode: RuntimeMode::Strict,
            a: vec![vec![1.0]],
            b: vec![1.0],
            assume_a: None,
            lower: None,
            transposed: None,
            check_finite: None,
            expected: LinalgExpectedOutcome::Vector {
                values: vec![1.0],
                atol: 1.0e-12,
                rtol: 1.0e-12,
                expect_warning_ill_conditioned: None,
            },
        };
        let oracle_case = OracleCaseOutput {
            case_id: "oracle_missing_solve".to_owned(),
            status: "error".to_owned(),
            result_kind: "unsupported_function".to_owned(),
            result: serde_json::json!({}),
            error: Some("unsupported function: solve".to_owned()),
        };
        let observed = Err(LinalgError::SingularMatrix);

        let (passed, message, max_diff, tolerance_used) =
            compare_linalg_case_against_oracle(&case, &oracle_case, &observed);

        assert!(!passed, "oracle error + rust error must fail closed");
        assert!(message.contains("unjudgeable"), "{message}");
        assert!(max_diff.is_none());
        assert!(tolerance_used.is_none());
    }

    #[test]
    fn stats_oracle_error_and_rust_error_is_not_an_auto_pass() {
        let case = StatsCase {
            case_id: "oracle_missing_stats".to_owned(),
            category: "summary".to_owned(),
            mode: RuntimeMode::Strict,
            function: "describe".to_owned(),
            args: vec![serde_json::json!([1.0, 2.0, 3.0])],
            expected: StatsExpected {
                kind: "scalar".to_owned(),
                value: Some(0.0),
                nobs: None,
                minmax: None,
                mean: None,
                variance: None,
                skewness: None,
                kurtosis: None,
                statistic: None,
                pvalue: None,
                slope: None,
                intercept: None,
                rvalue: None,
                stderr: None,
                array_value: None,
                atol: Some(1.0e-12),
                rtol: Some(1.0e-12),
                contract_ref: String::new(),
            },
        };
        let oracle_case = OracleCaseOutput {
            case_id: "oracle_missing_stats".to_owned(),
            status: "error".to_owned(),
            result_kind: "unsupported_function".to_owned(),
            result: serde_json::json!({}),
            error: Some("unsupported function: describe".to_owned()),
        };
        let observed = StatsObserved::Error("describe unavailable".to_owned());

        let (passed, message, max_diff, tolerance_used) =
            compare_stats_case_against_oracle(&case, &oracle_case, &observed);

        assert!(!passed, "oracle error + rust error must fail closed");
        assert!(message.contains("unjudgeable"), "{message}");
        assert!(max_diff.is_none());
        assert!(tolerance_used.is_none());
    }

    #[test]
    fn differential_test_fft_hardened_audit_emission() {
        let unique = format!("fsci-conformance-fft-audit-{}", super::now_unix_ms());
        let root = std::env::temp_dir().join(unique);
        fs::create_dir_all(&root).expect("create temp fixture root");
        let fixture_path = root.join("FSCI-P2C-005_fft_audit.json");
        let fixture = serde_json::json!({
            "packet_id": "P2C-005-AUDIT",
            "family": "fft_core",
            "cases": [{
                "case_id": "irfft_hardened_length_mismatch",
                "category": "irfft",
                "mode": "Hardened",
                "transform": "irfft",
                "normalization": "backward",
                "real_input": null,
                "complex_input": [[1.0, 0.0], [0.0, 0.0]],
                "output_len": 8,
                "sample_spacing": null,
                "shape": null,
                "expected": {
                    "kind": "error",
                    "error": "length mismatch"
                }
            }]
        });
        fs::write(
            &fixture_path,
            serde_json::to_vec_pretty(&fixture).expect("serialize fft audit fixture"),
        )
        .expect("write fft audit fixture");

        let oracle = default_test_oracle();
        let report =
            run_differential_test(&fixture_path, &oracle).expect("differential fft audit runs");

        assert_eq!(report.packet_id, "P2C-005-AUDIT");
        assert_eq!(report.fail_count, 0);
        assert_eq!(report.pass_count, 1);

        let audit_path =
            super::differential_audit_ledger_path_for_fixture(&fixture_path, &report.packet_id);
        let audit_jsonl = fs::read_to_string(&audit_path).expect("read fft audit ledger");
        assert!(audit_path.exists());
        assert!(audit_jsonl.contains("\"kind\":\"mode_decision\""));
        assert!(audit_jsonl.contains("\"kind\":\"fail_closed\""));
        assert!(audit_jsonl.contains("length_mismatch"));
    }

    #[test]
    fn differential_fft_quota_and_structured_logs() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-005_fft_core.json");
        let raw = fs::read_to_string(&fixture_path).expect("read fft fixture");
        let fixture: FftPacketFixture = serde_json::from_str(&raw).expect("parse fft fixture");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle).expect("fft differential");

        let mut by_case = std::collections::BTreeMap::new();
        for case in &fixture.cases {
            by_case.insert(case.case_id().to_owned(), case);
        }

        let mut case_count = 0usize;
        let mut logs = Vec::with_capacity(report.per_case_results.len());

        for case_result in &report.per_case_results {
            let case = by_case
                .get(&case_result.case_id)
                .expect("every report case should exist in fixture");
            case_count += 1;

            let log = StructuredCaseLog {
                test_id: case_result.case_id.clone(),
                category: case.category.clone(),
                input_summary: fft_case_input_summary(case),
                expected: fft_case_expected_summary(&case.expected),
                actual: case_result.message.clone(),
                diff: case_result.max_diff,
                tolerance: case_result.tolerance_used.clone(),
                pass: case_result.passed,
            };

            let payload =
                serde_json::to_string(&log).expect("structured conformance log should serialize");
            let parsed: serde_json::Value =
                serde_json::from_str(&payload).expect("structured log should parse");
            assert!(parsed.get("test_id").is_some());
            assert!(parsed.get("category").is_some());
            assert!(parsed.get("input_summary").is_some());
            assert!(parsed.get("expected").is_some());
            assert!(parsed.get("actual").is_some());
            assert!(parsed.get("diff").is_some());
            assert!(parsed.get("tolerance").is_some());
            assert!(parsed.get("pass").is_some());
            logs.push(log);
        }

        let expected_case_count = fixture.cases.len();
        assert!(
            expected_case_count >= 56,
            "fft fixture quota floor regressed: got {expected_case_count}"
        );
        assert_eq!(
            case_count, expected_case_count,
            "expected {expected_case_count} cases, got {case_count}"
        );
        assert_eq!(report.fail_count, 0);

        let output_dir = HarnessConfig::default_paths()
            .artifact_dir_for("P2C-005")
            .join("differential");
        fs::create_dir_all(&output_dir).expect("create fft differential artifact directory");
        let output_path = output_dir.join("structured_case_logs.json");
        let payload = serde_json::to_vec_pretty(&logs).expect("serialize fft logs");
        fs::write(&output_path, payload).expect("write fft structured logs");
        assert!(output_path.exists());
    }

    #[test]
    fn differential_test_graceful_oracle_missing() {
        let oracle = DifferentialOracleConfig {
            python_path: PathBuf::from("/nonexistent/python"),
            script_path: PathBuf::from("/nonexistent/script.py"),
            timeout_secs: 5,
            required: false,
        };
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-001_validate_tol.json");

        let report = run_differential_test(&fixture_path, &oracle)
            .expect("should succeed even with missing oracle");

        // Report should still pass against embedded expected values
        assert_eq!(report.fail_count, 0);
        assert!(matches!(report.oracle_status, OracleStatus::Missing { .. }));
    }

    #[test]
    fn differential_test_unknown_family_errors() {
        let unique = format!("fsci-diff-unknown-{}", super::now_unix_ms());
        let dir = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&dir).expect("create temp dir");
        let fixture_path = dir.join("bad_fixture.json");
        fs::write(
            &fixture_path,
            r#"{"packet_id":"FSCI-P2C-999","family":"unknown_type","cases":[]}"#,
        )
        .expect("write fixture");

        let oracle = default_test_oracle();
        let result = run_differential_test(&fixture_path, &oracle);
        assert!(result.is_err(), "unknown family should produce an error");
    }

    #[test]
    fn differential_case_runner_catches_panics_and_continues() {
        let oracle_status = OracleStatus::Available;
        let results = [
            super::run_case_with_panic_capture("case-1", &oracle_status, None, || {
                (true, "ok".to_owned(), Some(0.0), None)
            }),
            super::run_case_with_panic_capture("case-2", &oracle_status, None, || {
                std::panic::panic_any("boom")
            }),
            super::run_case_with_panic_capture("case-3", &oracle_status, None, || {
                (true, "still running".to_owned(), None, None)
            }),
        ];

        assert!(results[0].passed);
        assert!(!results[1].passed);
        assert!(results[1].message.starts_with("PANIC: boom"));
        assert!(results[2].passed);
    }

    #[test]
    fn differential_case_runner_clears_poisoned_audit_ledgers_after_panic() {
        let oracle_status = OracleStatus::Available;
        let audit_ledger = std::sync::Mutex::new(super::AuditLedger::new());

        let result = super::run_case_with_panic_capture(
            "case-poison",
            &oracle_status,
            Some(&audit_ledger),
            || {
                let _guard = audit_ledger.lock().expect("lock");
                std::panic::panic_any("audit poison");
            },
        );

        assert!(!result.passed);
        assert!(result.message.starts_with("PANIC: audit poison"));
        assert!(audit_ledger.lock().is_ok(), "poison should be cleared");
    }

    #[test]
    fn differential_audit_ledger_recovery_clears_preexisting_poison() {
        let audit_ledger = std::sync::Mutex::new(super::AuditLedger::new());

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = audit_ledger.lock().expect("lock");
            std::panic::panic_any("preexisting audit poison");
        }));

        assert!(
            audit_ledger.lock().is_err(),
            "setup should poison the audit ledger"
        );
        {
            let _guard = super::recover_sync_audit_ledger(&audit_ledger);
        }

        assert!(
            audit_ledger.lock().is_ok(),
            "recovery should clear stale poison state"
        );
    }

    #[test]
    fn conformance_report_serializes_to_json() {
        let fixture_path = HarnessConfig::default_paths()
            .fixture_root
            .join("FSCI-P2C-002_linalg_core.json");
        let oracle = default_test_oracle();
        let report = run_differential_test(&fixture_path, &oracle).expect("test should succeed");

        let json = serde_json::to_string_pretty(&report).expect("report must serialize");
        let parsed: ConformanceReport =
            serde_json::from_str(&json).expect("report must round-trip");
        assert_eq!(parsed.packet_id, report.packet_id);
        assert_eq!(parsed.pass_count, report.pass_count);
        assert_eq!(parsed.per_case_results.len(), report.per_case_results.len());
    }

    #[test]
    fn differential_parity_artifacts_preserve_rich_case_data() {
        let unique = format!("fsci-diff-artifacts-{}", super::now_unix_ms());
        let fixture_root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&fixture_root).expect("create temp dir");
        let fixture_path = fixture_root.join("FSCI-P2C-777_stats.json");
        fs::write(&fixture_path, "{}").expect("write placeholder fixture");

        let report = ConformanceReport {
            fixture_path: fixture_path.display().to_string(),
            packet_id: "FSCI-P2C-777".to_owned(),
            family: "stats".to_owned(),
            pass_count: 1,
            fail_count: 0,
            oracle_status: OracleStatus::Available,
            per_case_results: vec![DifferentialCaseResult {
                case_id: "case-1".to_owned(),
                passed: true,
                message: "scalar match".to_owned(),
                max_diff: Some(1.0e-12),
                tolerance_used: Some(ToleranceUsed {
                    atol: 1.0e-12,
                    rtol: 1.0e-10,
                    comparison_mode: "scalar".to_owned(),
                }),
                oracle_status: OracleStatus::Available,
            }],
            generated_unix_ms: 42,
        };

        let artifacts = write_differential_parity_artifacts(&fixture_path, &report)
            .expect("rich differential artifacts should persist");
        let raw = fs::read_to_string(&artifacts.report_path).expect("read persisted report");
        let parsed: PacketReport =
            serde_json::from_str(&raw).expect("packet report should round-trip");

        assert_eq!(parsed.schema_version, super::packet_report_schema_v2());
        assert_eq!(
            parsed.fixture_path.as_deref(),
            Some(report.fixture_path.as_str())
        );
        assert_eq!(parsed.oracle_status, Some(OracleStatus::Available));
        assert_eq!(parsed.passed_cases, report.pass_count);
        assert_eq!(parsed.failed_cases, report.fail_count);
        assert_eq!(parsed.case_results.len(), report.per_case_results.len());
        assert_eq!(
            parsed.differential_case_results.as_ref(),
            Some(&report.per_case_results)
        );
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn drift_diff_artifact_surfaces_failed_cases() {
        let unique = format!("fsci-drift-diff-{}", super::now_unix_ms());
        let fixture_root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&fixture_root).expect("create temp dir");
        let fixture_path = fixture_root.join("FSCI-P2C-888_stats.json");
        fs::write(&fixture_path, "{}").expect("write placeholder fixture");

        let report = ConformanceReport {
            fixture_path: fixture_path.display().to_string(),
            packet_id: "FSCI-P2C-888".to_owned(),
            family: "stats_core".to_owned(),
            pass_count: 1,
            fail_count: 1,
            oracle_status: OracleStatus::Available,
            per_case_results: vec![
                DifferentialCaseResult {
                    case_id: "ok".to_owned(),
                    passed: true,
                    message: "matched".to_owned(),
                    max_diff: Some(0.0),
                    tolerance_used: None,
                    oracle_status: OracleStatus::Available,
                },
                DifferentialCaseResult {
                    case_id: "drift".to_owned(),
                    passed: false,
                    message: "max diff exceeded tolerance".to_owned(),
                    max_diff: Some(1.0e-3),
                    tolerance_used: Some(ToleranceUsed {
                        atol: 1.0e-6,
                        rtol: 1.0e-6,
                        comparison_mode: "scalar".to_owned(),
                    }),
                    oracle_status: OracleStatus::Available,
                },
            ],
            generated_unix_ms: 123,
        };

        let drift = build_drift_diff_report(&report);
        assert_eq!(drift.total_cases, 2);
        assert_eq!(drift.drifted_cases, 1);
        assert_eq!(drift.cases[1].case_id, "drift");
        assert_eq!(drift.cases[1].max_diff, Some(1.0e-3));

        let drift_path =
            write_drift_diff_artifact(&fixture_path, &drift).expect("write drift artifact");
        let raw = fs::read_to_string(&drift_path).expect("read drift artifact");
        let parsed: super::DriftDiffReport =
            serde_json::from_str(&raw).expect("parse drift artifact");
        assert_eq!(parsed, drift);
    }

    #[test]
    fn live_oracle_capture_lane_discovers_p2c_fixtures_and_writes_drift() {
        let unique = format!("fsci-live-oracle-lane-{}", super::now_unix_ms());
        let fixture_root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(&fixture_root).expect("create temp fixture root");
        let fixture_path = fixture_root.join("FSCI-P2C-001_validate_tol.json");
        let fixture = serde_json::json!({
            "packet_id": "FSCI-P2C-001",
            "family": "integrate.validate_tol",
            "cases": [{
                "case_id": "scalar_rtol_clamp",
                "mode": "Strict",
                "n": 3,
                "rtol": {"kind": "scalar", "value": 1e-30},
                "atol": {"kind": "scalar", "value": 1e-8},
                "expected": {
                    "kind": "ok",
                    "rtol": {"kind": "scalar", "value": 2.220446049250313e-14},
                    "atol": {"kind": "scalar", "value": 1e-8},
                    "warning_rtol_clamped": true
                }
            }]
        });
        fs::write(
            &fixture_path,
            serde_json::to_vec_pretty(&fixture).expect("serialize fixture"),
        )
        .expect("write fixture");
        fs::write(fixture_root.join("smoke_case.json"), "{}").expect("write ignored smoke file");

        let config = HarnessConfig {
            oracle_root: PathBuf::from("/unused/oracle-root"),
            fixture_root: fixture_root.clone(),
            strict_mode: true,
        };
        let oracle = DifferentialOracleConfig {
            python_path: PathBuf::from("/nonexistent/python"),
            script_path: PathBuf::from("/nonexistent/script.py"),
            timeout_secs: 1,
            required: false,
        };

        let discovered =
            discover_differential_fixture_paths(&config).expect("discover differential fixtures");
        assert_eq!(discovered, vec![fixture_path.clone()]);

        let report = run_live_oracle_capture_lane(&config, &oracle, Some("P2C-001"))
            .expect("live oracle lane should tolerate optional missing oracle");
        assert_eq!(report.total_packets, 1);
        assert_eq!(report.total_cases, 1);
        assert_eq!(report.drifted_cases, 0);
        assert_eq!(report.oracle_unavailable_packets, 1);
        assert!(
            fixture_root
                .join("artifacts/FSCI-P2C-001/differential/drift_diff.json")
                .exists()
        );
    }

    #[test]
    fn drift_gate_blocks_packet_above_zero_threshold() {
        let report = LiveOracleCaptureReport {
            schema_version: 1,
            generated_unix_ms: 123,
            fixture_root: "fixtures".to_owned(),
            total_packets: 1,
            total_cases: 2,
            drifted_cases: 1,
            oracle_available_packets: 1,
            oracle_unavailable_packets: 0,
            packets: vec![DriftDiffReport {
                schema_version: 1,
                packet_id: "FSCI-P2C-012".to_owned(),
                family: "stats_core".to_owned(),
                fixture_path: "FSCI-P2C-012_stats_core.json".to_owned(),
                generated_unix_ms: 123,
                oracle_status: OracleStatus::Available,
                total_cases: 2,
                drifted_cases: 1,
                cases: Vec::new(),
            }],
        };

        let gate = evaluate_drift_gate(&report, &default_zero_drift_thresholds());

        assert!(!gate.passed());
        assert_eq!(gate.failed_packets, 1);
        assert_eq!(gate.total_drifted_cases, 1);
        assert_eq!(gate.packets[0].max_allowed_drifted_cases, 0);
    }

    #[test]
    fn zero_drift_oracle_coverage_blocks_all_missing_packets() {
        let report = LiveOracleCaptureReport {
            schema_version: 1,
            generated_unix_ms: 789,
            fixture_root: "fixtures".to_owned(),
            total_packets: 2,
            total_cases: 2,
            drifted_cases: 0,
            oracle_available_packets: 0,
            oracle_unavailable_packets: 2,
            packets: vec![
                DriftDiffReport {
                    schema_version: 1,
                    packet_id: "FSCI-P2C-011".to_owned(),
                    family: "signal_core".to_owned(),
                    fixture_path: "FSCI-P2C-011_signal_core.json".to_owned(),
                    generated_unix_ms: 789,
                    oracle_status: OracleStatus::Missing {
                        reason: "python missing".to_owned(),
                    },
                    total_cases: 1,
                    drifted_cases: 0,
                    cases: Vec::new(),
                },
                DriftDiffReport {
                    schema_version: 1,
                    packet_id: "FSCI-P2C-012".to_owned(),
                    family: "stats_core".to_owned(),
                    fixture_path: "FSCI-P2C-012_stats_core.json".to_owned(),
                    generated_unix_ms: 789,
                    oracle_status: OracleStatus::Missing {
                        reason: "scipy missing".to_owned(),
                    },
                    total_cases: 1,
                    drifted_cases: 0,
                    cases: Vec::new(),
                },
            ],
        };

        let message = validate_zero_drift_oracle_coverage(&report)
            .expect_err("all-missing oracle coverage must fail the zero-drift gate");

        assert!(message.contains("0 oracle-backed packet(s)"));
        assert!(message.contains("below required minimum 1"));
    }

    #[test]
    fn zero_drift_oracle_coverage_allows_partial_oracle_availability() {
        let report = LiveOracleCaptureReport {
            schema_version: 1,
            generated_unix_ms: 790,
            fixture_root: "fixtures".to_owned(),
            total_packets: 2,
            total_cases: 2,
            drifted_cases: 0,
            oracle_available_packets: 1,
            oracle_unavailable_packets: 1,
            packets: vec![
                DriftDiffReport {
                    schema_version: 1,
                    packet_id: "FSCI-P2C-011".to_owned(),
                    family: "signal_core".to_owned(),
                    fixture_path: "FSCI-P2C-011_signal_core.json".to_owned(),
                    generated_unix_ms: 790,
                    oracle_status: OracleStatus::Available,
                    total_cases: 1,
                    drifted_cases: 0,
                    cases: Vec::new(),
                },
                DriftDiffReport {
                    schema_version: 1,
                    packet_id: "FSCI-P2C-012".to_owned(),
                    family: "stats_core".to_owned(),
                    fixture_path: "FSCI-P2C-012_stats_core.json".to_owned(),
                    generated_unix_ms: 790,
                    oracle_status: OracleStatus::Missing {
                        reason: "scipy missing".to_owned(),
                    },
                    total_cases: 1,
                    drifted_cases: 0,
                    cases: Vec::new(),
                },
            ],
        };

        validate_zero_drift_oracle_coverage(&report)
            .expect("one oracle-backed packet keeps the drift gate meaningful");
    }

    #[test]
    fn drift_gate_honors_packet_specific_threshold() {
        let report = LiveOracleCaptureReport {
            schema_version: 1,
            generated_unix_ms: 456,
            fixture_root: "fixtures".to_owned(),
            total_packets: 1,
            total_cases: 2,
            drifted_cases: 1,
            oracle_available_packets: 1,
            oracle_unavailable_packets: 0,
            packets: vec![DriftDiffReport {
                schema_version: 1,
                packet_id: "FSCI-P2C-011".to_owned(),
                family: "signal_core".to_owned(),
                fixture_path: "FSCI-P2C-011_signal_core.json".to_owned(),
                generated_unix_ms: 456,
                oracle_status: OracleStatus::Available,
                total_cases: 2,
                drifted_cases: 1,
                cases: Vec::new(),
            }],
        };
        let thresholds = [DriftGateThreshold {
            packet_id: "FSCI-P2C-011".to_owned(),
            max_drifted_cases: 1,
        }];

        let gate = evaluate_drift_gate(&report, &thresholds);

        assert!(gate.passed());
        assert_eq!(gate.failed_packets, 0);
        assert_eq!(gate.packets[0].max_allowed_drifted_cases, 1);
    }

    // ═══════════════════════════════════════════════════════════════
    // Packet runner registry tests (§bd-3jh.19.10)
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn packet_family_all_has_17_entries() {
        assert_eq!(PacketFamily::ALL.len(), 17);
    }

    #[test]
    fn packet_family_ids_are_unique() {
        let mut ids: Vec<&str> = PacketFamily::ALL.iter().map(|f| f.packet_id()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), PacketFamily::ALL.len());
    }

    #[test]
    fn packet_family_names_are_unique() {
        let mut names: Vec<&str> = PacketFamily::ALL.iter().map(|f| f.family_name()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), PacketFamily::ALL.len());
    }

    #[test]
    fn packet_family_from_family_str_roundtrip() {
        for family in PacketFamily::ALL {
            let detected = PacketFamily::from_family_str(family.family_name());
            assert_eq!(
                detected,
                Some(family),
                "from_family_str should roundtrip for {}",
                family.family_name()
            );
        }
    }

    #[test]
    fn packet_family_from_family_str_unknown() {
        assert_eq!(PacketFamily::from_family_str("completely_unknown"), None);
    }

    #[test]
    fn discover_fixtures_finds_existing() {
        let config = HarnessConfig::default_paths();
        let fixtures = discover_fixtures(&config).expect("discover fixtures");
        assert!(
            fixtures.len() >= 3,
            "should find at least P2C-001, P2C-002, and P2C-003 fixtures"
        );
        let families: Vec<PacketFamily> = fixtures.iter().map(|(f, _)| *f).collect();
        assert!(families.contains(&PacketFamily::ValidateTol));
        assert!(families.contains(&PacketFamily::LinalgCore));
        assert!(families.contains(&PacketFamily::Optimize));
        assert!(families.contains(&PacketFamily::InterpolateCore));
        assert!(families.contains(&PacketFamily::NdimageCore));
        assert!(families.contains(&PacketFamily::ConstantsCore));
        assert!(families.contains(&PacketFamily::IoCore));
    }

    #[test]
    fn ensure_artifact_layout_creates_dirs() {
        let unique = format!("fsci-layout-{}", super::now_unix_ms());
        let root = PathBuf::from("/tmp").join(unique);
        fs::create_dir_all(root.join("fixtures")).expect("create root");
        let config = HarnessConfig {
            oracle_root: PathBuf::new(),
            fixture_root: root.join("fixtures"),
            strict_mode: true,
        };

        let base = ensure_artifact_layout(&config, "P2C-008").expect("create layout");
        assert!(base.join("anchor").exists());
        assert!(base.join("contracts").exists());
        assert!(base.join("threats").exists());

        // Idempotent: calling again should succeed
        ensure_artifact_layout(&config, "P2C-008").expect("idempotent layout");
    }

    #[test]
    fn aggregate_report_sums_correctly() {
        let make_cases = |n: usize, pass: bool| -> Vec<super::CaseResult> {
            (0..n)
                .map(|i| super::CaseResult {
                    case_id: format!("case_{i}"),
                    passed: pass,
                    message: String::new(),
                })
                .collect()
        };

        let mut cases_1 = make_cases(10, true);
        cases_1.extend(make_cases(2, false));
        let mut cases_2 = make_cases(20, true);
        cases_2.extend(make_cases(1, false));

        let reports = vec![
            super::PacketReport {
                schema_version: super::packet_report_schema_v1(),
                packet_id: "FSCI-P2C-001".to_owned(),
                family: "validate_tol".to_owned(),
                case_results: cases_1,
                passed_cases: 10,
                failed_cases: 2,
                fixture_path: None,
                oracle_status: None,
                differential_case_results: None,
                report_kind: super::ReportKind::Unspecified,
                generated_unix_ms: 0,
            },
            super::PacketReport {
                schema_version: super::packet_report_schema_v1(),
                packet_id: "FSCI-P2C-002".to_owned(),
                family: "linalg_core".to_owned(),
                case_results: cases_2,
                passed_cases: 20,
                failed_cases: 1,
                fixture_path: None,
                oracle_status: None,
                differential_case_results: None,
                report_kind: super::ReportKind::Unspecified,
                generated_unix_ms: 0,
            },
        ];

        let agg = aggregate_packet_reports(&reports);
        assert_eq!(agg.total_cases, 33);
        assert_eq!(agg.total_passed, 30);
        assert_eq!(agg.total_failed, 3);
        assert_eq!(agg.packets.len(), 2);
    }

    #[test]
    fn aggregate_report_serializes() {
        let agg = AggregateParityReport {
            generated_unix_ms: 1234567890,
            packets: vec![],
            total_cases: 0,
            total_passed: 0,
            total_failed: 0,
        };
        let json = serde_json::to_string(&agg).expect("serialize");
        let parsed: AggregateParityReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.generated_unix_ms, 1234567890);
    }

    #[test]
    fn packet_family_fixture_filename() {
        assert_eq!(
            PacketFamily::ValidateTol.fixture_filename(),
            "FSCI-P2C-001_validate_tol.json"
        );
        assert_eq!(
            PacketFamily::RuntimeCasp.fixture_filename(),
            "FSCI-P2C-008_runtime_casp.json"
        );
        assert_eq!(
            PacketFamily::NdimageCore.fixture_filename(),
            "FSCI-P2C-015_ndimage_core.json"
        );
        assert_eq!(
            PacketFamily::ConstantsCore.fixture_filename(),
            "FSCI-P2C-016_constants_core.json"
        );
        assert_eq!(
            PacketFamily::IoCore.fixture_filename(),
            "FSCI-P2C-017_io_core.json"
        );
    }

    #[test]
    fn packet_family_has_runner() {
        assert!(PacketFamily::ValidateTol.has_runner());
        assert!(PacketFamily::LinalgCore.has_runner());
        assert!(PacketFamily::Optimize.has_runner());
        assert!(PacketFamily::SparseOps.has_runner());
        assert!(PacketFamily::Fft.has_runner());
        assert!(PacketFamily::Special.has_runner());
        assert!(PacketFamily::ArrayApi.has_runner());
        assert!(PacketFamily::RuntimeCasp.has_runner());
        assert!(PacketFamily::ConstantsCore.has_runner());
        assert!(PacketFamily::InterpolateCore.has_runner());
        assert!(PacketFamily::NdimageCore.has_runner());
        assert!(PacketFamily::IoCore.has_runner());
    }

    #[test]
    fn fixture_backed_p2c_packets_are_registered() -> Result<(), Box<dyn std::error::Error>> {
        let config = HarnessConfig::default_paths();
        let mut fixture_packet_ids = Vec::new();
        for entry in fs::read_dir(&config.fixture_root)? {
            let path = entry?.path();
            if path.extension().and_then(std::ffi::OsStr::to_str) != Some("json") {
                continue;
            }
            if let Some(name) = path.file_name().and_then(std::ffi::OsStr::to_str) {
                if !name.starts_with("FSCI-P2C-") {
                    continue;
                }
                if let Some(packet_id) = name.get(..12) {
                    fixture_packet_ids.push(packet_id.to_owned());
                }
            }
        }
        fixture_packet_ids.sort();
        fixture_packet_ids.dedup();

        let mut registry_packet_ids: Vec<String> = PacketFamily::ALL
            .iter()
            .map(|family| family.packet_id().to_owned())
            .collect();
        registry_packet_ids.sort();

        assert_eq!(
            fixture_packet_ids, registry_packet_ids,
            "every fixture-backed P2C packet must be in PacketFamily::ALL"
        );
        Ok(())
    }

    #[test]
    fn compare_unordered_points_handles_greedy_trap() {
        // br-cwgm: greedy matching would assign A→X first, leaving B with
        // only the out-of-tol Y, but optimal matches A→Y and B→X. Hungarian
        // (augmenting paths) must find the optimal matching.
        let expected = vec![
            vec![0.0, 0.0], // A
            vec![0.4, 0.0], // B
        ];
        let got = vec![
            vec![0.1, 0.0], // X: diff 0.1 from A, diff 0.3 from B
            vec![0.5, 0.0], // Y: diff 0.5 from A (within tol), diff 1.1 from B (out of tol)
        ];
        // atol = 0.6 so A is within tol of both X and Y; B is within tol of X only.
        let r = super::compare_unordered_points(&got, &expected, 0.6, 0.0, "cwgm-test");
        assert!(r.is_ok(), "must find optimal matching, got err: {r:?}");
    }

    #[test]
    fn classify_python_stderr_buckets_missing_modules() {
        use super::{HarnessError, classify_python_stderr};
        // br-wbzg: scipy missing → PythonSciPyMissing (legacy variant).
        let err = classify_python_stderr(
            "python3".into(),
            "ModuleNotFoundError: No module named 'scipy'".into(),
        );
        assert!(matches!(err, HarnessError::PythonSciPyMissing { .. }));

        // numpy missing → PythonModuleMissing with module="numpy".
        let err = classify_python_stderr(
            "python3".into(),
            "ModuleNotFoundError: No module named 'numpy'".into(),
        );
        assert!(matches!(
            err,
            HarnessError::PythonModuleMissing { ref module, .. } if module == "numpy"
        ));

        // Unquoted form (older Python / ImportError path).
        let err = classify_python_stderr(
            "python3".into(),
            "ImportError: No module named sklearn.metrics".into(),
        );
        assert!(matches!(
            err,
            HarnessError::PythonModuleMissing { ref module, .. } if module == "sklearn"
        ));

        // Non-module failure falls through to PythonFailed.
        let err = classify_python_stderr(
            "python3".into(),
            "AttributeError: module scipy has no attribute btdtr".into(),
        );
        assert!(matches!(err, HarnessError::PythonFailed { .. }));
    }

    #[test]
    fn canonicalize_labels_renames_permuted_ids() {
        use super::canonicalize_labels_first_occurrence as canon;
        // Same partition, different IDs → same canonical form.
        let a = canon(&[0, 1, 1, 2, 0]);
        let b = canon(&[1, 2, 2, 0, 1]);
        assert_eq!(
            a, b,
            "br-7eaq: permuted cluster IDs must canonicalize equal"
        );

        // Different partition → different canonical form.
        let c = canon(&[0, 1, 0, 1, 0]);
        assert_ne!(
            a, c,
            "br-7eaq: different partitions must not canonicalize equal"
        );

        // First-occurrence ordering: first distinct ID seen maps to 0.
        assert_eq!(canon(&[5, 5, 7, 7, 5]), vec![0, 0, 1, 1, 0]);
    }
}
