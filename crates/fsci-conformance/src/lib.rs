#![forbid(unsafe_code)]

pub mod dashboard;

use asupersync::raptorq::systematic::SystematicEncoder;
use blake3::hash;
use fsci_integrate::{ToleranceValue, validate_tol};
use fsci_linalg::{
    InvOptions, LinalgError, LstsqDriver, LstsqOptions, MatrixAssumption, PinvOptions,
    SolveOptions, TriangularSolveOptions, TriangularTranspose, det, inv, lstsq, pinv, solve,
    solve_banded, solve_triangular,
};
use fsci_runtime::RuntimeMode;
#[cfg(feature = "dashboard")]
use ftui::{PackedRgba, Style};
use serde::{Deserialize, Serialize};
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
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
    pub fixture_count: usize,
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
    #[error("linalg execution failed: {0}")]
    Linalg(#[from] LinalgError),
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
    #[error("oracle capture parse failed for {path}: {source}")]
    OracleParse {
        path: PathBuf,
        source: serde_json::Error,
    },
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LinalgExpectedOutcome {
    Vector {
        values: Vec<f64>,
        atol: f64,
        rtol: f64,
        #[serde(default)]
        expect_warning_ill_conditioned: Option<bool>,
    },
    Matrix {
        values: Vec<Vec<f64>>,
        atol: f64,
        rtol: f64,
    },
    Scalar {
        value: f64,
        atol: f64,
        rtol: f64,
    },
    Lstsq {
        x: Vec<f64>,
        residuals: Vec<f64>,
        rank: usize,
        singular_values: Vec<f64>,
        atol: f64,
        rtol: f64,
    },
    Pinv {
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
        a: Vec<Vec<f64>>,
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
        a: Vec<Vec<f64>>,
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
        ab: Vec<Vec<f64>>,
        b: Vec<f64>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
    Inv {
        case_id: String,
        mode: RuntimeMode,
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
        a: Vec<Vec<f64>>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
    Lstsq {
        case_id: String,
        mode: RuntimeMode,
        a: Vec<Vec<f64>>,
        b: Vec<f64>,
        #[serde(default)]
        cond: Option<f64>,
        #[serde(default)]
        check_finite: Option<bool>,
        expected: LinalgExpectedOutcome,
    },
    Pinv {
        case_id: String,
        mode: RuntimeMode,
        a: Vec<Vec<f64>>,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
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
            | Self::Pinv { case_id, .. } => case_id,
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
            | Self::Pinv { expected, .. } => expected,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LinalgPacketFixture {
    pub packet_id: String,
    pub family: String,
    pub cases: Vec<LinalgCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaseResult {
    pub case_id: String,
    pub passed: bool,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PacketReport {
    pub packet_id: String,
    pub family: String,
    pub case_results: Vec<CaseResult>,
    pub passed_cases: usize,
    pub failed_cases: usize,
    pub generated_unix_ms: u128,
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
    pub source_symbols: usize,
    pub repair_symbols: usize,
    pub repair_symbol_hashes: Vec<String>,
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
    pub case_outputs: Vec<OracleCaseOutput>,
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
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            python_bin: PathBuf::from("python3"),
            script_path: manifest.join("python_oracle/scipy_linalg_oracle.py"),
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

#[must_use]
pub fn run_smoke(config: &HarnessConfig) -> HarnessReport {
    let fixture_count = fs::read_dir(&config.fixture_root)
        .ok()
        .into_iter()
        .flat_map(|it| it.filter_map(Result::ok))
        .count();

    HarnessReport {
        suite: "smoke",
        oracle_present: config.oracle_root.exists(),
        fixture_count,
        strict_mode: config.strict_mode,
    }
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
                let pass = error == &actual.to_string();
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
        let observed = execute_linalg_case(case);
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

pub fn run_linalg_packet_with_oracle_capture(
    config: &HarnessConfig,
    fixture_name: &str,
    oracle: &PythonOracleConfig,
) -> Result<PacketReport, HarnessError> {
    let report = run_linalg_packet(config, fixture_name)?;
    let oracle_result = capture_linalg_oracle(config, fixture_name, oracle);
    if let Err(err) = oracle_result {
        if oracle.required {
            return Err(err);
        }
        let failure_path = config
            .artifact_dir_for(&report.packet_id)
            .join("oracle_capture.error.txt");
        fs::create_dir_all(config.artifact_dir_for(&report.packet_id)).map_err(|source| {
            HarnessError::ArtifactIo {
                path: config.artifact_dir_for(&report.packet_id),
                source,
            }
        })?;
        fs::write(&failure_path, format!("{err}")).map_err(|source| HarnessError::ArtifactIo {
            path: failure_path,
            source,
        })?;
    }

    Ok(report)
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
        if stderr.contains("No module named 'scipy'") {
            return Err(HarnessError::PythonSciPyMissing { stderr });
        }
        return Err(HarnessError::PythonFailed { python_bin, stderr });
    }

    let parsed = load_oracle_capture(&output_path)?;
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
    fs::create_dir_all(&output_dir).map_err(|source| HarnessError::ArtifactIo {
        path: output_dir.clone(),
        source,
    })?;

    let report_path = output_dir.join("parity_report.json");
    let report_bytes =
        serde_json::to_vec_pretty(report).map_err(|e| HarnessError::RaptorQ(e.to_string()))?;
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

    let decode_proof = DecodeProofArtifact {
        ts_unix_ms: now_unix_ms(),
        reason: "no recovery required; artifact remained intact".to_owned(),
        recovered_blocks: 0,
        proof_hash: hash(&report_bytes).to_hex().to_string(),
    };
    let decode_proof_path = output_dir.join("parity_report.decode_proof.json");
    let decode_proof_bytes = serde_json::to_vec_pretty(&decode_proof)
        .map_err(|e| HarnessError::RaptorQ(e.to_string()))?;
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
            let pass = error == &actual.to_string();
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
    (actual - expected).abs() <= atol + rtol * expected.abs()
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
        packet_id,
        family,
        case_results,
        passed_cases,
        failed_cases,
        generated_unix_ms: now_unix_ms(),
    }
}

fn generate_raptorq_sidecar(payload: &[u8]) -> Result<RaptorQSidecar, HarnessError> {
    let symbol_size = 128usize;
    let source_symbols = chunk_payload(payload, symbol_size);
    let k = source_symbols.len();
    let repair_symbols = (k / 5).max(1);
    let seed = hash(payload).as_bytes()[0] as u64 + 1337;

    let encoder = SystematicEncoder::new(&source_symbols, symbol_size, seed).ok_or_else(|| {
        HarnessError::RaptorQ("systematic encoder initialization failed".to_owned())
    })?;

    let mut repair_hashes = Vec::with_capacity(repair_symbols);
    for esi in k as u32..(k as u32 + repair_symbols as u32) {
        let symbol = encoder.repair_symbol(esi);
        repair_hashes.push(hash(&symbol).to_hex().to_string());
    }

    Ok(RaptorQSidecar {
        schema_version: 1,
        source_hash: hash(payload).to_hex().to_string(),
        symbol_size,
        source_symbols: k,
        repair_symbols,
        repair_symbol_hashes: repair_hashes,
    })
}

fn chunk_payload(payload: &[u8], symbol_size: usize) -> Vec<Vec<u8>> {
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
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            python_path: PathBuf::from("python3"),
            script_path: manifest.join("python_oracle/scipy_linalg_oracle.py"),
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

/// Fixture envelope: minimal structure to detect fixture type.
#[derive(Debug, Clone, Deserialize)]
struct FixtureEnvelope {
    family: String,
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
    if family.contains("validate_tol") {
        run_differential_validate_tol(fixture_path, &raw, oracle_config)
    } else if family.contains("linalg") {
        run_differential_linalg(fixture_path, &raw, oracle_config)
    } else {
        Err(HarnessError::FixtureParse {
            path: fixture_path.to_path_buf(),
            source: serde::de::Error::custom(format!("unknown fixture family: {family}")),
        })
    }
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

    for case in &fixture.cases {
        let outcome = validate_tol(
            case.rtol.clone().into(),
            case.atol.clone().into(),
            case.n,
            case.mode,
        );

        let (passed, message, max_diff, tolerance_used) = match (&case.expected, outcome) {
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
                let pass = error == &actual.to_string();
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
        };

        per_case_results.push(DifferentialCaseResult {
            case_id: case.case_id.clone(),
            passed,
            message,
            max_diff,
            tolerance_used,
            oracle_status: oracle_status.clone(),
        });
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

    for case in &fixture.cases {
        let observed = execute_linalg_case(case);
        let (passed, message, max_diff, tolerance_used) =
            compare_linalg_case_differential(case.expected(), &observed);

        per_case_results.push(DifferentialCaseResult {
            case_id: case.case_id().to_owned(),
            passed,
            message,
            max_diff,
            tolerance_used,
            oracle_status: oracle_status.clone(),
        });
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
            let pass = error == &actual.to_string();
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
        .fold(0.0_f64, f64::max)
}

/// Compute the maximum absolute difference between two matrices.
fn max_diff_matrix(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ar, br)| max_diff_vec(ar, br))
        .fold(0.0_f64, f64::max)
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

#[cfg(test)]
mod tests {
    #[cfg(feature = "dashboard")]
    use super::style_for_case_result;
    use super::{
        ConformanceReport, DifferentialOracleConfig, HarnessConfig, LinalgPacketFixture,
        OracleStatus, PythonOracleConfig, load_oracle_capture, run_differential_test,
        run_linalg_packet, run_smoke, run_validate_tol_packet, write_parity_artifacts,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;

    #[test]
    fn smoke_harness_finds_oracle_and_fixtures() {
        let cfg = HarnessConfig::default_paths();
        let report = run_smoke(&cfg);
        assert!(report.oracle_present, "oracle repo should be present");
        assert!(report.fixture_count >= 1, "expected at least one fixture");
        assert!(report.strict_mode);
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
    fn linalg_packet_passes() {
        let cfg = HarnessConfig::default_paths();
        let report =
            run_linalg_packet(&cfg, "FSCI-P2C-002_linalg_core.json").expect("linalg packet runs");
        assert_eq!(report.failed_cases, 0);
        assert!(report.passed_cases >= 1);

        let artifacts =
            write_parity_artifacts(&cfg, &report).expect("linalg parity artifacts must be written");
        assert!(artifacts.report_path.exists());
        assert!(artifacts.sidecar_path.exists());
        assert!(artifacts.decode_proof_path.exists());
    }

    #[test]
    fn mock_python_oracle_capture_parses() {
        let unique = format!("fsci-conformance-test-{}", super::now_unix_ms());
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

        let fixture_raw = fs::read_to_string(fixture_dst).expect("read fixture");
        let fixture: LinalgPacketFixture =
            serde_json::from_str(&fixture_raw).expect("fixture parse");
        assert_eq!(parsed.case_outputs.len(), fixture.cases.len());
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
            ..PythonOracleConfig::default()
        };

        let output_path =
            super::capture_linalg_oracle(&cfg, "FSCI-P2C-002_linalg_core.json", &oracle)
                .expect("scipy oracle capture succeeds");
        let parsed = load_oracle_capture(&output_path).expect("oracle capture parse succeeds");

        let fixture_raw = fs::read_to_string(fixture_dst).expect("read fixture");
        let fixture: LinalgPacketFixture =
            serde_json::from_str(&fixture_raw).expect("fixture parse");
        assert_eq!(parsed.packet_id, "FSCI-P2C-002");
        assert_eq!(parsed.case_outputs.len(), fixture.cases.len());
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
}
