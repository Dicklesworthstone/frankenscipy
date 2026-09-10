#![forbid(unsafe_code)]

use crate::{
    DifferentialOracleConfig, HarnessConfig, differential_audit_ledger_path_for_fixture,
    run_differential_test,
};
use blake3::hash;
use fsci_runtime::AuditLedger;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct E2eOrchestratorConfig {
    pub artifact_root: PathBuf,
    pub fixture_root: PathBuf,
    pub packet_filter: Option<String>,
    pub scenario_filter: Option<String>,
    pub run_id: Option<String>,
}

impl Default for E2eOrchestratorConfig {
    fn default() -> Self {
        let harness = HarnessConfig::default_paths();
        Self {
            artifact_root: harness.fixture_root.join("artifacts"),
            fixture_root: harness.fixture_root,
            packet_filter: None,
            scenario_filter: None,
            run_id: None,
        }
    }
}

#[derive(Debug, Error)]
pub enum E2eOrchestratorError {
    #[error("artifact root does not exist: {path}")]
    ArtifactRootMissing { path: PathBuf },
    #[error("failed to read directory {path}: {source}")]
    ReadDir { path: PathBuf, source: io::Error },
    #[error("failed to read scenario {path}: {source}")]
    ScenarioRead { path: PathBuf, source: io::Error },
    #[error("failed to parse scenario {path}: {source}")]
    ScenarioParse {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("no e2e scenarios found in {artifact_root}")]
    NoScenariosFound { artifact_root: PathBuf },
    #[error("failed to write log bundle {path}: {source}")]
    LogWrite { path: PathBuf, source: io::Error },
    #[error("failed to serialize log entry: {0}")]
    LogSerialize(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Deserialize)]
pub struct E2eScenario {
    pub scenario_id: String,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub steps: Vec<E2eStep>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct E2eStep {
    pub step_name: String,
    #[serde(flatten)]
    pub action: E2eStepAction,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum E2eStepAction {
    CheckPathExists { path: String },
    RunDifferentialFixture { fixture: String },
}

#[derive(Debug, Clone)]
pub struct DiscoveredScenario {
    pub packet_id: String,
    pub packet_dir: PathBuf,
    pub scenario_path: PathBuf,
    pub scenario: E2eScenario,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct E2eRunSummary {
    pub run_id: String,
    pub total_scenarios: usize,
    pub passed_scenarios: usize,
    pub failed_scenarios: usize,
    pub scenario_results: Vec<ScenarioRunResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScenarioRunResult {
    pub packet_id: String,
    pub scenario_id: String,
    pub passed: bool,
    pub failed_step: Option<String>,
    pub replay_command: String,
    pub run_bundle_dir: PathBuf,
    pub events_path: PathBuf,
    pub summary_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InputHashEntry {
    pub path: String,
    pub blake3: Option<String>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EnvironmentSnapshot {
    pub rust_version: String,
    pub os: String,
    pub arch: String,
    pub cpu_count: usize,
    pub seed: u64,
    pub input_hashes: Vec<InputHashEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct E2eLogEntry {
    pub scenario_id: String,
    pub step_name: String,
    pub timestamp_ms: u128,
    pub duration_ms: u128,
    pub outcome: String,
    pub message: String,
    pub environment: EnvironmentSnapshot,
    pub artifact_refs: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay_command: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScenarioRunBundleSummary {
    pub packet_id: String,
    pub scenario_id: String,
    pub run_id: String,
    pub passed: bool,
    pub failed_step: Option<String>,
    pub replay_command: String,
    pub generated_unix_ms: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StepExecutionOutcome {
    passed: bool,
    message: String,
    artifact_refs: Vec<String>,
}

pub fn discover_scenarios(
    config: &E2eOrchestratorConfig,
) -> Result<Vec<DiscoveredScenario>, E2eOrchestratorError> {
    if !config.artifact_root.exists() {
        return Err(E2eOrchestratorError::ArtifactRootMissing {
            path: config.artifact_root.clone(),
        });
    }

    let mut packet_dirs = Vec::new();
    for entry in
        fs::read_dir(&config.artifact_root).map_err(|source| E2eOrchestratorError::ReadDir {
            path: config.artifact_root.clone(),
            source,
        })?
    {
        let entry = entry.map_err(|source| E2eOrchestratorError::ReadDir {
            path: config.artifact_root.clone(),
            source,
        })?;
        let path = entry.path();
        if path.is_dir() {
            packet_dirs.push(path);
        }
    }

    packet_dirs.sort();

    let mut scenarios = Vec::new();
    for packet_dir in packet_dirs {
        let packet_id = packet_dir
            .file_name()
            .and_then(|name| name.to_str())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| String::from("unknown-packet"));

        if !matches_filter(&packet_id, config.packet_filter.as_deref()) {
            continue;
        }

        let scenario_paths = discover_packet_scenario_files(&packet_dir)?;
        if scenario_paths.is_empty() {
            continue;
        }

        for scenario_path in scenario_paths {
            let raw = fs::read_to_string(&scenario_path).map_err(|source| {
                E2eOrchestratorError::ScenarioRead {
                    path: scenario_path.clone(),
                    source,
                }
            })?;
            let scenario: E2eScenario = serde_json::from_str(&raw).map_err(|source| {
                E2eOrchestratorError::ScenarioParse {
                    path: scenario_path.clone(),
                    source,
                }
            })?;

            if !matches_filter(&scenario.scenario_id, config.scenario_filter.as_deref()) {
                continue;
            }

            scenarios.push(DiscoveredScenario {
                packet_id: packet_id.clone(),
                packet_dir: packet_dir.clone(),
                scenario_path,
                scenario,
            });
        }
    }

    scenarios.sort_by(|a, b| {
        a.packet_id
            .cmp(&b.packet_id)
            .then_with(|| a.scenario.scenario_id.cmp(&b.scenario.scenario_id))
    });

    Ok(scenarios)
}

fn discover_packet_scenario_files(packet_dir: &Path) -> Result<Vec<PathBuf>, E2eOrchestratorError> {
    let e2e_dir = packet_dir.join("e2e");
    if !e2e_dir.exists() || !e2e_dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut scenario_files = Vec::new();
    let nested_scenarios_dir = e2e_dir.join("scenarios");
    if nested_scenarios_dir.exists() && nested_scenarios_dir.is_dir() {
        scenario_files.extend(discover_json_files(&nested_scenarios_dir)?);
    }

    for entry in fs::read_dir(&e2e_dir).map_err(|source| E2eOrchestratorError::ReadDir {
        path: e2e_dir.clone(),
        source,
    })? {
        let entry = entry.map_err(|source| E2eOrchestratorError::ReadDir {
            path: e2e_dir.clone(),
            source,
        })?;
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
        {
            scenario_files.push(path);
        }
    }

    scenario_files.sort();
    scenario_files.dedup();
    Ok(scenario_files)
}

pub fn run_orchestrator(
    config: &E2eOrchestratorConfig,
) -> Result<E2eRunSummary, E2eOrchestratorError> {
    let discovered = discover_scenarios(config)?;
    if discovered.is_empty() {
        return Err(E2eOrchestratorError::NoScenariosFound {
            artifact_root: config.artifact_root.clone(),
        });
    }

    let run_id = config.run_id.clone().unwrap_or_else(generate_run_id);
    let mut scenario_results = Vec::with_capacity(discovered.len());
    for scenario in &discovered {
        scenario_results.push(run_single_scenario(config, scenario, &run_id)?);
    }

    let passed_scenarios = scenario_results
        .iter()
        .filter(|result| result.passed)
        .count();
    let total_scenarios = scenario_results.len();
    Ok(E2eRunSummary {
        run_id,
        total_scenarios,
        passed_scenarios,
        failed_scenarios: total_scenarios.saturating_sub(passed_scenarios),
        scenario_results,
    })
}

fn run_single_scenario(
    config: &E2eOrchestratorConfig,
    discovered: &DiscoveredScenario,
    run_id: &str,
) -> Result<ScenarioRunResult, E2eOrchestratorError> {
    let scenario_id = discovered.scenario.scenario_id.clone();
    let scenario_dir_name = sanitize_path_component(&scenario_id);
    let run_bundle_dir = discovered
        .packet_dir
        .join("e2e")
        .join("runs")
        .join(run_id)
        .join(scenario_dir_name);
    fs::create_dir_all(&run_bundle_dir).map_err(|source| E2eOrchestratorError::LogWrite {
        path: run_bundle_dir.clone(),
        source,
    })?;

    let events_path = run_bundle_dir.join("events.jsonl");
    let events_file =
        File::create(&events_path).map_err(|source| E2eOrchestratorError::LogWrite {
            path: events_path.clone(),
            source,
        })?;
    let mut writer = BufWriter::new(events_file);

    let replay_command = build_replay_command(config, &discovered.packet_id, &scenario_id);
    let seed = discovered.scenario.seed.unwrap_or_else(|| {
        hash(format!("{run_id}:{scenario_id}").as_bytes()).as_bytes()[0..8]
            .iter()
            .fold(0_u64, |acc, byte| (acc << 8) | u64::from(*byte))
    });
    let environment = collect_environment_snapshot(config, discovered, seed);

    write_log_entry(
        &mut writer,
        &events_path,
        E2eLogEntry {
            scenario_id: scenario_id.clone(),
            step_name: String::from("setup"),
            timestamp_ms: now_unix_ms(),
            duration_ms: 0,
            outcome: String::from("pass"),
            message: String::from("orchestrator setup complete"),
            environment: environment.clone(),
            artifact_refs: vec![discovered.scenario_path.display().to_string()],
            replay_command: None,
        },
    )?;
    write_log_entry(
        &mut writer,
        &events_path,
        E2eLogEntry {
            scenario_id: scenario_id.clone(),
            step_name: String::from("pre-scenario"),
            timestamp_ms: now_unix_ms(),
            duration_ms: 0,
            outcome: String::from("pass"),
            message: String::from("pre-scenario hook complete"),
            environment: environment.clone(),
            artifact_refs: Vec::new(),
            replay_command: None,
        },
    )?;

    let mut scenario_passed = true;
    let mut failed_step = None;

    for step in &discovered.scenario.steps {
        let started = Instant::now();
        let step_outcome = execute_step(config, step);
        let duration_ms = started.elapsed().as_millis();

        if !step_outcome.passed {
            scenario_passed = false;
            failed_step = Some(step.step_name.clone());
        }

        write_log_entry(
            &mut writer,
            &events_path,
            E2eLogEntry {
                scenario_id: scenario_id.clone(),
                step_name: step.step_name.clone(),
                timestamp_ms: now_unix_ms(),
                duration_ms,
                outcome: if step_outcome.passed {
                    String::from("pass")
                } else {
                    String::from("fail")
                },
                message: step_outcome.message,
                environment: environment.clone(),
                artifact_refs: step_outcome.artifact_refs,
                replay_command: None,
            },
        )?;

        if !scenario_passed {
            break;
        }
    }

    write_log_entry(
        &mut writer,
        &events_path,
        E2eLogEntry {
            scenario_id: scenario_id.clone(),
            step_name: String::from("post-scenario"),
            timestamp_ms: now_unix_ms(),
            duration_ms: 0,
            outcome: if scenario_passed {
                String::from("pass")
            } else {
                String::from("fail")
            },
            message: if scenario_passed {
                String::from("post-scenario hook complete")
            } else {
                String::from("post-scenario hook complete after failure")
            },
            environment: environment.clone(),
            artifact_refs: Vec::new(),
            replay_command: None,
        },
    )?;
    write_log_entry(
        &mut writer,
        &events_path,
        E2eLogEntry {
            scenario_id: scenario_id.clone(),
            step_name: String::from("teardown"),
            timestamp_ms: now_unix_ms(),
            duration_ms: 0,
            outcome: String::from("pass"),
            message: String::from("teardown hook complete"),
            environment: environment.clone(),
            artifact_refs: Vec::new(),
            replay_command: None,
        },
    )?;

    if !scenario_passed {
        write_log_entry(
            &mut writer,
            &events_path,
            E2eLogEntry {
                scenario_id: scenario_id.clone(),
                step_name: String::from("replay_command"),
                timestamp_ms: now_unix_ms(),
                duration_ms: 0,
                outcome: String::from("fail"),
                message: replay_command.clone(),
                environment: environment.clone(),
                artifact_refs: Vec::new(),
                replay_command: Some(replay_command.clone()),
            },
        )?;
    }

    writer
        .flush()
        .map_err(|source| E2eOrchestratorError::LogWrite {
            path: events_path.clone(),
            source,
        })?;

    let summary_path = run_bundle_dir.join("summary.json");
    let bundle_summary = ScenarioRunBundleSummary {
        packet_id: discovered.packet_id.clone(),
        scenario_id: scenario_id.clone(),
        run_id: run_id.to_owned(),
        passed: scenario_passed,
        failed_step: failed_step.clone(),
        replay_command: replay_command.clone(),
        generated_unix_ms: now_unix_ms(),
    };
    let summary_json = serde_json::to_vec_pretty(&bundle_summary)?;
    fs::write(&summary_path, summary_json).map_err(|source| E2eOrchestratorError::LogWrite {
        path: summary_path.clone(),
        source,
    })?;

    Ok(ScenarioRunResult {
        packet_id: discovered.packet_id.clone(),
        scenario_id,
        passed: scenario_passed,
        failed_step,
        replay_command,
        run_bundle_dir,
        events_path,
        summary_path,
    })
}

fn execute_step(config: &E2eOrchestratorConfig, step: &E2eStep) -> StepExecutionOutcome {
    match &step.action {
        E2eStepAction::CheckPathExists { path } => {
            let resolved = resolve_from_root(&config.artifact_root, path);
            let exists = resolved.exists();
            StepExecutionOutcome {
                passed: exists,
                message: if exists {
                    format!("path exists: {}", resolved.display())
                } else {
                    format!("path missing: {}", resolved.display())
                },
                artifact_refs: vec![resolved.display().to_string()],
            }
        }
        E2eStepAction::RunDifferentialFixture { fixture } => {
            let fixture_path = resolve_from_root(&config.fixture_root, fixture);
            let oracle = DifferentialOracleConfig::default();
            match run_differential_test(&fixture_path, &oracle) {
                Ok(report) => {
                    let passed = report.fail_count == 0;
                    let mut artifact_refs = vec![fixture_path.display().to_string()];
                    let audit_ledger_path = differential_audit_ledger_path_for_fixture(
                        &fixture_path,
                        &report.packet_id,
                    );
                    if audit_ledger_path.exists() {
                        artifact_refs.push(audit_ledger_path.display().to_string());
                    }
                    StepExecutionOutcome {
                        passed,
                        message: format!(
                            "differential fixture run: pass_count={} fail_count={} oracle_status={:?}",
                            report.pass_count, report.fail_count, report.oracle_status
                        ),
                        artifact_refs,
                    }
                }
                Err(error) => StepExecutionOutcome {
                    passed: false,
                    message: format!("differential fixture failed: {error}"),
                    artifact_refs: vec![fixture_path.display().to_string()],
                },
            }
        }
    }
}

fn discover_json_files(root: &Path) -> Result<Vec<PathBuf>, E2eOrchestratorError> {
    let mut stack = vec![root.to_path_buf()];
    let mut json_files = Vec::new();

    while let Some(current) = stack.pop() {
        for entry in fs::read_dir(&current).map_err(|source| E2eOrchestratorError::ReadDir {
            path: current.clone(),
            source,
        })? {
            let entry = entry.map_err(|source| E2eOrchestratorError::ReadDir {
                path: current.clone(),
                source,
            })?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
            {
                json_files.push(path);
            }
        }
    }

    json_files.sort();
    Ok(json_files)
}

fn matches_filter(value: &str, filter: Option<&str>) -> bool {
    filter.is_none_or(|f| value.eq_ignore_ascii_case(f))
}

fn resolve_from_root(root: &Path, raw_path: &str) -> PathBuf {
    let candidate = PathBuf::from(raw_path);
    if candidate.is_absolute() {
        candidate
    } else {
        let direct = root.join(&candidate);
        if !direct.exists() {
            if let Some(stripped) = raw_path.strip_prefix("P2C-") {
                let canonical = root.join(format!("FSCI-P2C-{stripped}"));
                if canonical.exists() {
                    return canonical;
                }
            } else if let Some(stripped) = raw_path.strip_prefix("FSCI-P2C-") {
                let legacy = root.join(format!("P2C-{stripped}"));
                if legacy.exists() {
                    return legacy;
                }
            }
        }
        direct
    }
}

fn collect_environment_snapshot(
    config: &E2eOrchestratorConfig,
    discovered: &DiscoveredScenario,
    seed: u64,
) -> EnvironmentSnapshot {
    let mut input_paths = HashSet::new();
    input_paths.insert(discovered.scenario_path.clone());

    for step in &discovered.scenario.steps {
        match &step.action {
            E2eStepAction::CheckPathExists { path } => {
                input_paths.insert(resolve_from_root(&config.artifact_root, path));
            }
            E2eStepAction::RunDifferentialFixture { fixture } => {
                input_paths.insert(resolve_from_root(&config.fixture_root, fixture));
            }
        }
    }

    let mut input_hashes: Vec<InputHashEntry> = input_paths
        .into_iter()
        .map(|path| {
            if !path.exists() {
                return InputHashEntry {
                    path: path.display().to_string(),
                    blake3: None,
                    status: String::from("missing"),
                };
            }
            match fs::read(&path) {
                Ok(bytes) => InputHashEntry {
                    path: path.display().to_string(),
                    blake3: Some(hash(&bytes).to_hex().to_string()),
                    status: String::from("present"),
                },
                Err(_) => InputHashEntry {
                    path: path.display().to_string(),
                    blake3: None,
                    status: String::from("unreadable"),
                },
            }
        })
        .collect();
    input_hashes.sort_by(|a, b| a.path.cmp(&b.path));

    EnvironmentSnapshot {
        rust_version: rustc_version(),
        os: String::from(std::env::consts::OS),
        arch: String::from(std::env::consts::ARCH),
        cpu_count: std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1),
        seed,
        input_hashes,
    }
}

fn rustc_version() -> String {
    let output = Command::new("rustc").arg("--version").output();
    match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_owned(),
        Ok(out) => format!(
            "unavailable (status={} stderr={})",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ),
        Err(error) => format!("unavailable ({error})"),
    }
}

fn build_replay_command(
    config: &E2eOrchestratorConfig,
    packet_id: &str,
    scenario_id: &str,
) -> String {
    format!(
        "cargo run -p fsci-conformance --bin e2e_orchestrator -- --artifact-root {} --packet {} --scenario {}",
        shell_quote_posix(&config.artifact_root.display().to_string()),
        shell_quote_posix(packet_id),
        shell_quote_posix(scenario_id),
    )
}

/// POSIX single-quote a string so that copy-pasting the resulting
/// command into `sh`/`bash`/`zsh` cannot trigger command substitution,
/// parameter expansion, or meta-character interpretation. An embedded
/// single-quote is encoded as `'\''` (close, escape, reopen).
///
/// Empty strings become `''`.
///
/// Per br-y18q: prior code did bare `{}` interpolation of fixture-owned
/// packet_id / scenario_id, which would emit a runnable `'; rm -rf /'`
/// if a fixture ever contained shell metacharacters.
fn shell_quote_posix(s: &str) -> String {
    if s.is_empty() {
        return String::from("''");
    }
    // Fast path: plain alnum / dash / underscore / dot / slash / equals
    // needs no quoting.
    let needs_quote = s.chars().any(|c| {
        !(c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | '=' | ':' | '+'))
    });
    if !needs_quote {
        return s.to_owned();
    }
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        if c == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(c);
        }
    }
    out.push('\'');
    out
}

fn sanitize_path_component(value: &str) -> String {
    let mut cleaned = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            cleaned.push(ch);
        } else {
            cleaned.push('_');
        }
    }
    if cleaned.is_empty() {
        String::from("scenario")
    } else {
        cleaned
    }
}

fn generate_run_id() -> String {
    format!("run-{}", now_unix_ms())
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn write_log_entry(
    writer: &mut BufWriter<File>,
    events_path: &Path,
    entry: E2eLogEntry,
) -> Result<(), E2eOrchestratorError> {
    serde_json::to_writer(&mut *writer, &entry)?;
    writer
        .write_all(b"\n")
        .map_err(|source| E2eOrchestratorError::LogWrite {
            path: events_path.to_path_buf(),
            source,
        })
}

// ═══════════════════════════════════════════════════════════════════
// Audit Ledger Emission (§0.19)
// ═══════════════════════════════════════════════════════════════════

/// Write an AuditLedger to a .jsonl file in the specified directory.
///
/// The output file is named `audit_ledger.jsonl` and contains one JSON
/// object per line (JSON Lines format).
pub fn emit_audit_ledger(
    artifact_dir: &Path,
    ledger: &AuditLedger,
) -> Result<PathBuf, E2eOrchestratorError> {
    fs::create_dir_all(artifact_dir).map_err(|source| E2eOrchestratorError::LogWrite {
        path: artifact_dir.to_path_buf(),
        source,
    })?;

    let output_path = artifact_dir.join("audit_ledger.jsonl");
    let file = File::create(&output_path).map_err(|source| E2eOrchestratorError::LogWrite {
        path: output_path.clone(),
        source,
    })?;
    let mut writer = BufWriter::new(file);

    for entry in ledger.entries() {
        serde_json::to_writer(&mut writer, entry)?;
        writer
            .write_all(b"\n")
            .map_err(|source| E2eOrchestratorError::LogWrite {
                path: output_path.clone(),
                source,
            })?;
    }

    writer
        .flush()
        .map_err(|source| E2eOrchestratorError::LogWrite {
            path: output_path.clone(),
            source,
        })?;

    Ok(output_path)
}

/// Write an AuditLedger to a .jsonl file with a custom filename.
pub fn emit_audit_ledger_named(
    artifact_dir: &Path,
    filename: &str,
    ledger: &AuditLedger,
) -> Result<PathBuf, E2eOrchestratorError> {
    fs::create_dir_all(artifact_dir).map_err(|source| E2eOrchestratorError::LogWrite {
        path: artifact_dir.to_path_buf(),
        source,
    })?;

    let output_path = artifact_dir.join(filename);
    let file = File::create(&output_path).map_err(|source| E2eOrchestratorError::LogWrite {
        path: output_path.clone(),
        source,
    })?;
    let mut writer = BufWriter::new(file);

    for entry in ledger.entries() {
        serde_json::to_writer(&mut writer, entry)?;
        writer
            .write_all(b"\n")
            .map_err(|source| E2eOrchestratorError::LogWrite {
                path: output_path.clone(),
                source,
            })?;
    }

    writer
        .flush()
        .map_err(|source| E2eOrchestratorError::LogWrite {
            path: output_path.clone(),
            source,
        })?;

    Ok(output_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsci_runtime::{AuditAction, AuditEvent, casp_now_unix_ms};
    use std::io::BufRead;
    use tempfile::TempDir;

    #[test]
    fn shell_quote_posix_passes_through_safe_strings() {
        assert_eq!(shell_quote_posix("FSCI-P2C-002"), "FSCI-P2C-002");
        assert_eq!(shell_quote_posix("/tmp/artifacts"), "/tmp/artifacts");
        assert_eq!(shell_quote_posix("x=1"), "x=1");
    }

    #[test]
    fn shell_quote_posix_quotes_dangerous_input() {
        // br-y18q: a fixture scenario_id that looks like a command
        // injection MUST be rendered as a literal single-quoted string,
        // not a shell-interpreted fragment.
        assert_eq!(
            shell_quote_posix("scenario; rm -rf /tmp/x"),
            "'scenario; rm -rf /tmp/x'"
        );
        assert_eq!(shell_quote_posix("$(id)"), "'$(id)'");
        assert_eq!(shell_quote_posix("`id`"), "'`id`'");
        assert_eq!(shell_quote_posix("a b"), "'a b'");
    }

    #[test]
    fn shell_quote_posix_escapes_embedded_quote() {
        assert_eq!(shell_quote_posix("a'b"), r"'a'\''b'");
    }

    #[test]
    fn shell_quote_posix_empty_is_empty_quotes() {
        assert_eq!(shell_quote_posix(""), "''");
    }

    #[test]
    fn emit_audit_ledger_writes_jsonl() {
        let tmp = TempDir::new().expect("create temp dir");
        let mut ledger = AuditLedger::new();

        ledger.record(AuditEvent::new(
            casp_now_unix_ms(),
            AuditLedger::fingerprint_bytes(b"test1"),
            AuditAction::FailClosed {
                reason: "test_fail".to_string(),
            },
            "rejected",
        ));
        ledger.record(AuditEvent::new(
            casp_now_unix_ms(),
            AuditLedger::fingerprint_bytes(b"test2"),
            AuditAction::BoundedRecovery {
                recovery_action: "test_recovery".to_string(),
            },
            "recovered",
        ));

        let output_path = emit_audit_ledger(tmp.path(), &ledger).expect("emit works");

        assert!(output_path.exists());
        assert_eq!(output_path.file_name().unwrap(), "audit_ledger.jsonl");

        let file = File::open(&output_path).expect("open output");
        let reader = io::BufReader::new(file);
        let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();

        assert_eq!(lines.len(), 2, "should have 2 entries");
        assert!(lines[0].contains("test_fail"));
        assert!(lines[1].contains("test_recovery"));
    }

    #[test]
    fn emit_audit_ledger_named_uses_custom_filename() {
        let tmp = TempDir::new().expect("create temp dir");
        let ledger = AuditLedger::new();

        let output_path =
            emit_audit_ledger_named(tmp.path(), "custom_audit.jsonl", &ledger).expect("emit works");

        assert!(output_path.exists());
        assert_eq!(output_path.file_name().unwrap(), "custom_audit.jsonl");
    }

    #[test]
    fn emit_audit_ledger_creates_directory() {
        let tmp = TempDir::new().expect("create temp dir");
        let nested_dir = tmp.path().join("nested").join("artifact").join("dir");
        let ledger = AuditLedger::new();

        let output_path = emit_audit_ledger(&nested_dir, &ledger).expect("emit works");

        assert!(output_path.exists());
        assert!(nested_dir.exists());
    }
}
