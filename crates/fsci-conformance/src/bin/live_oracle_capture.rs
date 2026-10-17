#![forbid(unsafe_code)]

use fsci_conformance::{
    DifferentialOracleConfig, HarnessConfig, LiveOracleCaptureReport,
    default_zero_drift_thresholds, evaluate_drift_gate, generate_decode_proof_artifact,
    generate_raptorq_sidecar, run_live_oracle_capture_lane, validate_zero_drift_oracle_coverage,
};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Debug, Clone)]
struct CliArgs {
    fixture_root: PathBuf,
    oracle_root: PathBuf,
    python_path: PathBuf,
    timeout_secs: u64,
    packet_filter: Option<String>,
    output_path: Option<PathBuf>,
    allow_missing_oracle: bool,
    allow_drift: bool,
}

#[derive(Debug, Clone)]
enum CliParseError {
    Help,
    Message(String),
}

fn parse_cli_args(args: &[String]) -> Result<CliArgs, CliParseError> {
    let defaults = HarnessConfig::default_paths();
    let oracle_defaults = DifferentialOracleConfig::default();
    let mut fixture_root = defaults.fixture_root;
    let mut oracle_root = defaults.oracle_root;
    let mut python_path = oracle_defaults.python_path;
    let mut timeout_secs = oracle_defaults.timeout_secs;
    let mut packet_filter = None;
    let mut output_path = None;
    let mut allow_missing_oracle = false;
    let mut allow_drift = false;

    let mut index = 0;
    while let Some(arg) = args.get(index) {
        match arg.as_str() {
            "-h" | "--help" => return Err(CliParseError::Help),
            "--fixture-root" => {
                fixture_root = next_path(args, index, "--fixture-root")?;
                index += 2;
            }
            "--oracle-root" => {
                oracle_root = next_path(args, index, "--oracle-root")?;
                index += 2;
            }
            "--python" | "--python-path" => {
                python_path = next_path(args, index, "--python")?;
                index += 2;
            }
            "--timeout-secs" => {
                timeout_secs =
                    next_value(args, index, "--timeout-secs")?
                        .parse()
                        .map_err(|error| {
                            CliParseError::Message(format!("invalid --timeout-secs: {error}"))
                        })?;
                index += 2;
            }
            "--packet" | "--packet-filter" => {
                packet_filter = Some(next_value(args, index, "--packet")?.to_owned());
                index += 2;
            }
            "--output" => {
                output_path = Some(next_path(args, index, "--output")?);
                index += 2;
            }
            "--allow-missing-oracle" => {
                allow_missing_oracle = true;
                index += 1;
            }
            "--allow-drift" => {
                allow_drift = true;
                index += 1;
            }
            unknown => {
                return Err(CliParseError::Message(format!(
                    "unrecognized argument `{unknown}`"
                )));
            }
        }
    }

    Ok(CliArgs {
        fixture_root,
        oracle_root,
        python_path,
        timeout_secs,
        packet_filter,
        output_path,
        allow_missing_oracle,
        allow_drift,
    })
}

fn next_value<'a>(args: &'a [String], index: usize, flag: &str) -> Result<&'a str, CliParseError> {
    args.get(index + 1)
        .map(String::as_str)
        .ok_or_else(|| CliParseError::Message(format!("missing value for {flag}")))
}

fn next_path(args: &[String], index: usize, flag: &str) -> Result<PathBuf, CliParseError> {
    next_value(args, index, flag).map(PathBuf::from)
}

fn print_usage(program: &str) {
    eprintln!(
        "Usage: {program} [--fixture-root <path>] [--oracle-root <path>] [--python <path>] [--packet <filter>] [--output <path>] [--timeout-secs <n>] [--allow-missing-oracle] [--allow-drift]"
    );
    eprintln!(
        "When --output is set, also writes <output>.raptorq.json and <output>.decode_proof.json."
    );
}

fn sidecar_path_for(output_path: &Path) -> PathBuf {
    let mut path = output_path.as_os_str().to_os_string();
    path.push(".raptorq.json");
    PathBuf::from(path)
}

fn decode_proof_path_for(output_path: &Path) -> PathBuf {
    let mut path = output_path.as_os_str().to_os_string();
    path.push(".decode_proof.json");
    PathBuf::from(path)
}

fn write_sidecar_artifacts(output_path: &Path, report_bytes: &[u8]) -> Result<(), ExitCode> {
    let sidecar = match generate_raptorq_sidecar(report_bytes) {
        Ok(sidecar) => sidecar,
        Err(error) => {
            eprintln!("failed to generate RaptorQ sidecar: {error}");
            return Err(ExitCode::from(2));
        }
    };
    let sidecar_bytes = match serde_json::to_vec_pretty(&sidecar) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to serialize RaptorQ sidecar: {error}");
            return Err(ExitCode::from(2));
        }
    };
    let sidecar_path = sidecar_path_for(output_path);
    if let Err(error) = std::fs::write(&sidecar_path, sidecar_bytes) {
        eprintln!("failed to write {}: {error}", sidecar_path.display());
        return Err(ExitCode::from(2));
    }

    let decode_proof = match generate_decode_proof_artifact(report_bytes, &sidecar) {
        Ok(proof) => proof,
        Err(error) => {
            eprintln!("failed to generate decode proof: {error}");
            return Err(ExitCode::from(2));
        }
    };
    let decode_proof_bytes = match serde_json::to_vec_pretty(&decode_proof) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to serialize decode proof: {error}");
            return Err(ExitCode::from(2));
        }
    };
    let decode_proof_path = decode_proof_path_for(output_path);
    if let Err(error) = std::fs::write(&decode_proof_path, decode_proof_bytes) {
        eprintln!("failed to write {}: {error}", decode_proof_path.display());
        return Err(ExitCode::from(2));
    }

    Ok(())
}

fn write_summary(
    report: &LiveOracleCaptureReport,
    output_path: Option<&Path>,
) -> Result<(), ExitCode> {
    let bytes = match serde_json::to_vec_pretty(report) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to serialize live oracle report: {error}");
            return Err(ExitCode::from(2));
        }
    };

    if let Some(path) = output_path {
        if let Err(error) = std::fs::write(path, &bytes) {
            eprintln!("failed to write {}: {error}", path.display());
            return Err(ExitCode::from(2));
        }
        write_sidecar_artifacts(path, &bytes)?;
    } else {
        println!("{}", String::from_utf8_lossy(&bytes));
    }

    Ok(())
}

fn enforce_exit_policy(
    report: &LiveOracleCaptureReport,
    allow_missing_oracle: bool,
    allow_drift: bool,
) -> Result<(), ExitCode> {
    if report.total_packets == 0
        && let Err(message) = validate_zero_drift_oracle_coverage(report)
    {
        eprintln!("{message}");
        return Err(ExitCode::from(1));
    }
    if !allow_missing_oracle && report.oracle_unavailable_packets > 0 {
        eprintln!(
            "oracle unavailable for {} packet(s)",
            report.oracle_unavailable_packets
        );
        return Err(ExitCode::from(1));
    }
    if allow_drift {
        return Ok(());
    }

    let must_validate_oracle_coverage =
        !allow_missing_oracle || report.oracle_available_packets > 0 || report.total_packets == 0;
    if must_validate_oracle_coverage
        && let Err(message) = validate_zero_drift_oracle_coverage(report)
    {
        eprintln!("{message}");
        return Err(ExitCode::from(1));
    }

    let drift_gate = evaluate_drift_gate(report, &default_zero_drift_thresholds());
    if !drift_gate.passed() {
        eprintln!(
            "detected {} drifted case(s) above threshold across {} packet(s)",
            drift_gate.total_drifted_cases, drift_gate.failed_packets
        );
        for packet in drift_gate.packets.iter().filter(|packet| !packet.passed) {
            eprintln!(
                "{}: {} drifted case(s), max allowed {}",
                packet.packet_id, packet.drifted_cases, packet.max_allowed_drifted_cases
            );
        }
        return Err(ExitCode::from(1));
    }
    Ok(())
}

fn main() -> ExitCode {
    let argv = std::env::args().collect::<Vec<_>>();
    let program = argv
        .first()
        .cloned()
        .unwrap_or_else(|| String::from("live_oracle_capture"));
    let arg_slice = argv.get(1..).unwrap_or(&[]);
    let args = match parse_cli_args(arg_slice) {
        Ok(args) => args,
        Err(CliParseError::Help) => {
            print_usage(&program);
            return ExitCode::SUCCESS;
        }
        Err(CliParseError::Message(message)) => {
            eprintln!("{message}");
            print_usage(&program);
            return ExitCode::from(2);
        }
    };

    let harness = HarnessConfig {
        oracle_root: args.oracle_root,
        fixture_root: args.fixture_root,
        strict_mode: true,
    };
    let allow_missing_oracle = args.allow_missing_oracle;
    let allow_drift = args.allow_drift;
    let oracle = DifferentialOracleConfig {
        python_path: args.python_path,
        script_path: DifferentialOracleConfig::default().script_path,
        timeout_secs: args.timeout_secs,
        required: !allow_missing_oracle,
    };

    let report =
        match run_live_oracle_capture_lane(&harness, &oracle, args.packet_filter.as_deref()) {
            Ok(report) => report,
            Err(error) => {
                eprintln!("live oracle capture failed: {error}");
                return ExitCode::from(2);
            }
        };

    if let Err(code) = write_summary(&report, args.output_path.as_deref()) {
        return code;
    }
    if let Err(code) = enforce_exit_policy(&report, allow_missing_oracle, allow_drift) {
        return code;
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::{decode_proof_path_for, enforce_exit_policy, sidecar_path_for, write_summary};
    use fsci_conformance::{
        DecodeProofArtifact, DriftDiffReport, LiveOracleCaptureReport, OracleStatus, RaptorQSidecar,
    };
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn report_with_packet(
        oracle_status: OracleStatus,
        oracle_available_packets: usize,
        oracle_unavailable_packets: usize,
        drifted_cases: usize,
    ) -> LiveOracleCaptureReport {
        let oracle_compared_cases = usize::from(matches!(oracle_status, OracleStatus::Available));
        LiveOracleCaptureReport {
            schema_version: 1,
            generated_unix_ms: 123,
            fixture_root: "fixtures".to_owned(),
            total_packets: 1,
            total_cases: 1,
            drifted_cases,
            oracle_available_packets,
            oracle_compared_packets: oracle_available_packets,
            oracle_unavailable_packets,
            packets: vec![DriftDiffReport {
                schema_version: 1,
                packet_id: "FSCI-P2C-011".to_owned(),
                family: "signal_core".to_owned(),
                fixture_path: "FSCI-P2C-011_signal_core.json".to_owned(),
                generated_unix_ms: 123,
                oracle_status,
                oracle_compared_cases,
                total_cases: 1,
                drifted_cases,
                cases: Vec::new(),
            }],
        }
    }

    #[test]
    fn allow_missing_oracle_permits_all_missing_packets_without_drift() {
        let report = report_with_packet(
            OracleStatus::Missing {
                reason: "scipy not installed".to_owned(),
            },
            0,
            1,
            0,
        );
        assert!(enforce_exit_policy(&report, true, false).is_ok());
    }

    #[test]
    fn default_policy_rejects_all_missing_packets() {
        let report = report_with_packet(
            OracleStatus::Missing {
                reason: "scipy not installed".to_owned(),
            },
            0,
            1,
            0,
        );
        assert!(enforce_exit_policy(&report, false, false).is_err());
    }

    #[test]
    fn allow_missing_oracle_still_rejects_oracle_backed_drift() {
        let report = report_with_packet(OracleStatus::Available, 1, 0, 1);
        assert!(enforce_exit_policy(&report, true, false).is_err());
    }

    #[test]
    fn allow_missing_oracle_rejects_empty_capture() {
        let report = LiveOracleCaptureReport {
            schema_version: 1,
            generated_unix_ms: 123,
            fixture_root: "fixtures".to_owned(),
            total_packets: 0,
            total_cases: 0,
            drifted_cases: 0,
            oracle_available_packets: 0,
            oracle_compared_packets: 0,
            oracle_unavailable_packets: 0,
            packets: Vec::new(),
        };
        assert!(enforce_exit_policy(&report, true, false).is_err());
    }

    #[test]
    fn allow_drift_rejects_empty_capture() {
        let report = LiveOracleCaptureReport {
            schema_version: 1,
            generated_unix_ms: 123,
            fixture_root: "fixtures".to_owned(),
            total_packets: 0,
            total_cases: 0,
            drifted_cases: 0,
            oracle_available_packets: 0,
            oracle_compared_packets: 0,
            oracle_unavailable_packets: 0,
            packets: Vec::new(),
        };
        assert!(enforce_exit_policy(&report, true, true).is_err());
    }

    #[test]
    fn write_summary_emits_raptorq_sidecars_for_output_path()
    -> Result<(), Box<dyn std::error::Error>> {
        let report = report_with_packet(OracleStatus::Available, 1, 0, 0);
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let path = std::env::temp_dir().join(format!(
            "fsci-live-oracle-capture-{}-{unique}.json",
            std::process::id()
        ));

        write_summary(&report, Some(&path))
            .map_err(|code| format!("write_summary failed with exit code {code:?}"))?;

        let sidecar_path = sidecar_path_for(&path);
        let decode_proof_path = decode_proof_path_for(&path);
        let sidecar: RaptorQSidecar = serde_json::from_slice(&fs::read(sidecar_path)?)?;
        let decode_proof: DecodeProofArtifact =
            serde_json::from_slice(&fs::read(decode_proof_path)?)?;
        assert_eq!(sidecar.source_hash, decode_proof.proof_hash);
        Ok(())
    }
}
