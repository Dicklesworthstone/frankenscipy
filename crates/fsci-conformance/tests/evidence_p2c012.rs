//! [FSCI-P2C-012-I] Final Evidence Pack for stats conformance.
//!
//! Produces a self-contained evidence bundle at:
//!   fixtures/artifacts/FSCI-P2C-012/evidence/
//!
//! Covers packet-runner parity, fixture-driven differential outcomes, and
//! stats-specific risk notes for descriptive statistics and inference helpers.

use blake3::hash;
use fsci_conformance::{
    ConformanceReport, DifferentialOracleConfig, HarnessConfig, OracleStatus, PacketReport,
    RaptorQSidecar, StatsPacketFixture, generate_raptorq_sidecar, run_differential_test,
    run_stats_packet, write_parity_artifacts,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

#[derive(Debug, Serialize)]
struct FixtureManifest {
    packet_id: String,
    family: String,
    generated_at: String,
    fixtures: Vec<FixtureEntry>,
}

#[derive(Debug, Serialize)]
struct FixtureEntry {
    fixture_id: String,
    category: String,
    function: String,
    argument_count: usize,
    contract_ref: String,
    atol: f64,
    rtol: f64,
}

#[derive(Debug, Serialize)]
struct ParityGateReport {
    packet_id: String,
    all_gates_pass: bool,
    runner_passed_cases: usize,
    runner_failed_cases: usize,
    differential_pass_count: usize,
    differential_fail_count: usize,
    oracle_status: OracleStatus,
    gates: Vec<ParityGate>,
}

#[derive(Debug, Serialize)]
struct ParityGate {
    fixture_id: String,
    category: String,
    function: String,
    pass: bool,
    max_abs_diff: f64,
    atol: f64,
    rtol: f64,
    comparison_mode: String,
    note: String,
}

#[derive(Debug, Serialize)]
struct ParityReport {
    packet_id: String,
    operation_summaries: Vec<OperationParitySummary>,
}

#[derive(Debug, Serialize)]
struct OperationParitySummary {
    function: String,
    total_fixtures: usize,
    passed: usize,
    failed: usize,
    max_abs_diff_across_all: f64,
}

#[derive(Debug, Serialize)]
struct RiskNotesReport {
    packet_id: String,
    notes: Vec<RiskNote>,
}

#[derive(Debug, Serialize)]
struct RiskNote {
    category: String,
    description: String,
    affected_operations: Vec<String>,
    mitigation: String,
}

#[derive(Debug, Serialize)]
struct EvidenceBundle {
    packet_id: String,
    generated_at: String,
    fixture_manifest: FixtureManifest,
    packet_report: PacketReport,
    differential_report: ConformanceReport,
    parity_gates: ParityGateReport,
    parity_report: ParityReport,
    risk_notes: RiskNotesReport,
    sidecar: Option<RaptorQSidecar>,
}

fn now_str() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("unix:{secs}")
}

fn fold_max(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0_f64, |acc, value| {
        if acc.is_nan() || value.is_nan() {
            f64::NAN
        } else {
            acc.max(value)
        }
    })
}

#[test]
fn evidence_p2c012_final_pack() -> Result<(), Box<dyn Error>> {
    let config = HarnessConfig::default_paths();
    let fixture_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/FSCI-P2C-012_stats_core.json");
    let evidence_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/FSCI-P2C-012/evidence");
    std::fs::create_dir_all(&evidence_dir)?;

    let fixture_raw = std::fs::read_to_string(&fixture_path)?;
    let fixture: StatsPacketFixture = serde_json::from_str(&fixture_raw)?;

    let fixture_manifest = FixtureManifest {
        packet_id: fixture.packet_id.clone(),
        family: fixture.family.clone(),
        generated_at: now_str(),
        fixtures: fixture
            .cases
            .iter()
            .map(|case| FixtureEntry {
                fixture_id: case.case_id.clone(),
                category: case.category.clone(),
                function: case.function.clone(),
                argument_count: case.args.len(),
                contract_ref: case.expected.contract_ref.clone(),
                atol: case.expected.atol.unwrap_or(1.0e-10),
                rtol: case.expected.rtol.unwrap_or(1.0e-10),
            })
            .collect(),
    };

    let packet_report = run_stats_packet(&config, "FSCI-P2C-012_stats_core.json")?;
    let parity_artifacts = write_parity_artifacts(&config, &packet_report)?;
    assert!(parity_artifacts.report_path.exists());
    assert!(parity_artifacts.sidecar_path.exists());
    assert!(parity_artifacts.decode_proof_path.exists());

    let oracle_config = DifferentialOracleConfig::default();
    let differential_report = run_differential_test(&fixture_path, &oracle_config)?;

    let case_entries: BTreeMap<&str, &FixtureEntry> = fixture_manifest
        .fixtures
        .iter()
        .map(|entry| (entry.fixture_id.as_str(), entry))
        .collect();

    let gates = differential_report
        .per_case_results
        .iter()
        .map(|result| -> Result<ParityGate, Box<dyn Error>> {
            let entry = case_entries
                .get(result.case_id.as_str())
                .copied()
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("missing fixture entry for {}", result.case_id),
                    )
                })?;
            let tolerance = result.tolerance_used.as_ref();
            Ok(ParityGate {
                fixture_id: entry.fixture_id.clone(),
                category: entry.category.clone(),
                function: entry.function.clone(),
                pass: result.passed,
                max_abs_diff: result.max_diff.unwrap_or(0.0),
                atol: tolerance.map_or(entry.atol, |value| value.atol),
                rtol: tolerance.map_or(entry.rtol, |value| value.rtol),
                comparison_mode: tolerance.map_or_else(
                    || String::from("mixed"),
                    |value| value.comparison_mode.clone(),
                ),
                note: result.message.clone(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    assert_eq!(
        gates.len(),
        fixture_manifest.fixtures.len(),
        "every stats fixture should have a parity gate"
    );

    let all_gates_pass = gates.iter().all(|gate| gate.pass)
        && packet_report.failed_cases == 0
        && differential_report.fail_count == 0;
    let parity_gates = ParityGateReport {
        packet_id: fixture.packet_id.clone(),
        all_gates_pass,
        runner_passed_cases: packet_report.passed_cases,
        runner_failed_cases: packet_report.failed_cases,
        differential_pass_count: differential_report.pass_count,
        differential_fail_count: differential_report.fail_count,
        oracle_status: differential_report.oracle_status.clone(),
        gates,
    };

    let mut grouped: BTreeMap<String, Vec<&ParityGate>> = BTreeMap::new();
    for gate in &parity_gates.gates {
        grouped.entry(gate.function.clone()).or_default().push(gate);
    }

    let parity_report = ParityReport {
        packet_id: fixture.packet_id.clone(),
        operation_summaries: grouped
            .into_iter()
            .map(|(function, gates)| OperationParitySummary {
                function,
                total_fixtures: gates.len(),
                passed: gates.iter().filter(|gate| gate.pass).count(),
                failed: gates.iter().filter(|gate| !gate.pass).count(),
                max_abs_diff_across_all: fold_max(gates.iter().map(|gate| gate.max_abs_diff)),
            })
            .collect(),
    };

    let risk_notes = RiskNotesReport {
        packet_id: fixture.packet_id.clone(),
        notes: vec![
            RiskNote {
                category: String::from("finite_sample_definition_drift"),
                description: String::from(
                    "Descriptive statistics and correlation helpers depend on exact sample-size conventions, divisor choices, and bias flags. Small implementation changes can silently shift results while still looking numerically plausible.",
                ),
                affected_operations: vec![
                    String::from("describe"),
                    String::from("skew"),
                    String::from("kurtosis"),
                    String::from("sem"),
                    String::from("variation"),
                ],
                mitigation: String::from(
                    "Keep fixture-derived parity gates at the function level and preserve explicit per-case tolerances instead of collapsing everything to a single suite-wide threshold.",
                ),
            },
            RiskNote {
                category: String::from("distribution_free_test_sensitivity"),
                description: String::from(
                    "Rank-based and goodness-of-fit tests are sensitive to sample ordering, ties, and tail behavior. Regressions often appear first in p-values rather than headline statistics.",
                ),
                affected_operations: vec![
                    String::from("ks_2samp"),
                    String::from("shapiro"),
                    String::from("ttest_1samp"),
                    String::from("pearsonr"),
                    String::from("spearmanr"),
                ],
                mitigation: String::from(
                    "Track both statistic and p-value parity in the evidence bundle and keep edge-case fixtures for tied, monotone, and boundary-distribution samples.",
                ),
            },
            RiskNote {
                category: String::from("normalization_and_entropy_guards"),
                description: String::from(
                    "Utilities like z-score normalization and entropy can degrade under degenerate or nearly degenerate inputs, producing NaNs or masking divide-by-zero behavior.",
                ),
                affected_operations: vec![
                    String::from("zscore"),
                    String::from("entropy"),
                    String::from("moment"),
                    String::from("iqr"),
                ],
                mitigation: String::from(
                    "Retain differential fixtures that cover narrow spreads and probability-style inputs, and fail the evidence pack if any previously finite path starts emitting non-finite outputs.",
                ),
            },
        ],
    };

    let evidence = EvidenceBundle {
        packet_id: fixture.packet_id.clone(),
        generated_at: now_str(),
        fixture_manifest,
        packet_report,
        differential_report,
        parity_gates,
        parity_report,
        risk_notes,
        sidecar: None,
    };

    let bundle_json = serde_json::to_string_pretty(&evidence)?;
    std::fs::write(evidence_dir.join("evidence_bundle.json"), &bundle_json)?;

    let sidecar = generate_raptorq_sidecar(bundle_json.as_bytes())?;
    std::fs::write(
        evidence_dir.join("evidence_bundle.raptorq.json"),
        serde_json::to_string_pretty(&sidecar)?,
    )?;

    std::fs::write(
        evidence_dir.join("fixture_manifest.json"),
        serde_json::to_string_pretty(&evidence.fixture_manifest)?,
    )?;
    std::fs::write(
        evidence_dir.join("runner_report.json"),
        serde_json::to_string_pretty(&evidence.packet_report)?,
    )?;
    std::fs::write(
        evidence_dir.join("differential_report.json"),
        serde_json::to_string_pretty(&evidence.differential_report)?,
    )?;
    std::fs::write(
        evidence_dir.join("parity_gates.json"),
        serde_json::to_string_pretty(&evidence.parity_gates)?,
    )?;
    std::fs::write(
        evidence_dir.join("parity_report.json"),
        serde_json::to_string_pretty(&evidence.parity_report)?,
    )?;
    std::fs::write(
        evidence_dir.join("risk_notes.json"),
        serde_json::to_string_pretty(&evidence.risk_notes)?,
    )?;
    std::fs::write(
        evidence_dir.join("evidence_bundle.blake3"),
        hash(bundle_json.as_bytes()).to_hex().to_string(),
    )?;

    for gate in &evidence.parity_gates.gates {
        let status = if gate.pass { "PASS" } else { "FAIL" };
        eprintln!(
            "  {status}: {} {} — {}",
            gate.function, gate.fixture_id, gate.note
        );
    }
    assert!(
        evidence.parity_gates.all_gates_pass,
        "all P2C-012 parity gates must pass"
    );
    Ok(())
}
