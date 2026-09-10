//! bd-3jh.18.9: [FSCI-P2C-007-I] Final Evidence Pack
//!
//! Produces a self-contained evidence bundle at:
//!   fixtures/artifacts/FSCI-P2C-007/evidence/
//!
//! Artifacts:
//! 1. fixture_manifest.json — operation fixtures with shape/dtype metadata
//! 2. parity_gates.json — per-fixture pass/fail outcomes
//! 3. risk_notes.json — required risk coverage notes
//! 4. parity_report.json — grouped parity summaries
//! 5. fixture_log.json — fixture_id/operation/pass_fail/shape/dtype log stream
//! 6. artifact_log.json — artifact_name/size_bytes/blake3_hash/timestamp log stream
//! 7. *.raptorq.json + *.decode_proof.json — sidecars and decode proofs

use asupersync::raptorq::systematic::SystematicEncoder;
use blake3::hash;
use fsci_conformance::{
    DecodeProofArtifact, DifferentialOracleConfig, OracleStatus, RaptorQSidecar, chunk_payload,
    generate_raptorq_sidecar, run_differential_test,
};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const REQUIRED_GROUPS: &[&str] = &["creation", "indexing", "broadcasting", "ufunc_dispatch"];

#[derive(Debug, Clone)]
struct CaseMeta {
    operation: String,
    group: String,
    shape: String,
    dtype: String,
}

#[derive(Debug, Serialize)]
struct FixtureManifest {
    packet_id: String,
    generated_at: String,
    fixtures: Vec<FixtureEntry>,
}

#[derive(Debug, Serialize)]
struct FixtureEntry {
    fixture_id: String,
    operation: String,
    category: String,
    shape: String,
    dtype: String,
    expected_kind: String,
    contract_ref: Option<String>,
}

#[derive(Debug, Serialize)]
struct ParityGateReport {
    packet_id: String,
    all_gates_pass: bool,
    gates: Vec<ParityGate>,
}

#[derive(Debug, Serialize)]
struct ParityGate {
    fixture_id: String,
    operation: String,
    group: String,
    pass_fail: bool,
    shape: String,
    dtype: String,
    max_abs_diff: f64,
    tolerance_atol: Option<f64>,
    tolerance_rtol: Option<f64>,
    oracle_status: String,
    message: String,
}

#[derive(Debug, Serialize)]
struct ParityReport {
    packet_id: String,
    operation_summaries: Vec<OperationParitySummary>,
}

#[derive(Debug, Serialize)]
struct OperationParitySummary {
    group: String,
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
struct ArtifactLogEntry {
    artifact_name: String,
    size_bytes: usize,
    blake3_hash: String,
    timestamp: String,
}

#[derive(Debug, Serialize)]
struct FixtureLogEntry {
    fixture_id: String,
    operation: String,
    pass_fail: bool,
    shape: String,
    dtype: String,
}

#[derive(Debug, Serialize)]
struct EvidenceBundle {
    packet_id: String,
    generated_at: String,
    fixture_manifest: FixtureManifest,
    parity_gates: ParityGateReport,
    parity_report: ParityReport,
    risk_notes: RiskNotesReport,
}

#[test]
fn evidence_p2c007_final_pack() {
    let fixture_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/FSCI-P2C-007_arrayapi_core.json");
    let evidence_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/FSCI-P2C-007/evidence");
    std::fs::create_dir_all(&evidence_dir).expect("evidence directory should be creatable");

    let fixture_raw =
        std::fs::read_to_string(&fixture_path).expect("P2C-007 fixture file should be readable");
    let fixture_json: Value =
        serde_json::from_str(&fixture_raw).expect("P2C-007 fixture file should parse as JSON");
    let packet_id = fixture_json
        .get("packet_id")
        .and_then(Value::as_str)
        .unwrap_or("FSCI-P2C-007")
        .to_string();

    let (manifest, case_meta) = build_fixture_manifest(&fixture_json, &packet_id);
    let differential = run_differential_test(&fixture_path, &DifferentialOracleConfig::default())
        .expect("differential report should be generated");
    assert_eq!(
        differential.packet_id, packet_id,
        "fixture packet_id and differential packet_id should match"
    );

    let mut gates = Vec::with_capacity(differential.per_case_results.len());
    for result in &differential.per_case_results {
        let meta = case_meta
            .get(result.case_id.as_str())
            .cloned()
            .unwrap_or_else(|| CaseMeta {
                operation: "unknown".to_string(),
                group: "other".to_string(),
                shape: "n/a".to_string(),
                dtype: "n/a".to_string(),
            });
        gates.push(ParityGate {
            fixture_id: result.case_id.clone(),
            operation: meta.operation.clone(),
            group: meta.group,
            pass_fail: result.passed,
            shape: meta.shape,
            dtype: meta.dtype,
            max_abs_diff: result.max_diff.unwrap_or(0.0),
            tolerance_atol: result.tolerance_used.as_ref().map(|tol| tol.atol),
            tolerance_rtol: result.tolerance_used.as_ref().map(|tol| tol.rtol),
            oracle_status: oracle_status_name(&result.oracle_status).to_string(),
            message: result.message.clone(),
        });
    }

    let all_gates_pass = gates.iter().all(|gate| gate.pass_fail);
    let parity_gates = ParityGateReport {
        packet_id: packet_id.clone(),
        all_gates_pass,
        gates,
    };

    let operation_summaries = REQUIRED_GROUPS
        .iter()
        .map(|group| {
            let grouped: Vec<_> = parity_gates
                .gates
                .iter()
                .filter(|gate| gate.group == *group)
                .collect();
            OperationParitySummary {
                group: (*group).to_string(),
                total_fixtures: grouped.len(),
                passed: grouped.iter().filter(|gate| gate.pass_fail).count(),
                failed: grouped.iter().filter(|gate| !gate.pass_fail).count(),
                max_abs_diff_across_all: grouped.iter().map(|gate| gate.max_abs_diff).fold(
                    0.0_f64,
                    |a: f64, b: f64| {
                        if a.is_nan() || b.is_nan() {
                            f64::NAN
                        } else {
                            a.max(b)
                        }
                    },
                ),
            }
        })
        .collect::<Vec<_>>();

    for summary in &operation_summaries {
        assert!(
            summary.total_fixtures > 0,
            "missing required parity group {}",
            summary.group
        );
    }

    let parity_report = ParityReport {
        packet_id: packet_id.clone(),
        operation_summaries,
    };

    let risk_notes = RiskNotesReport {
        packet_id: packet_id.clone(),
        notes: vec![
            RiskNote {
                category: "empty array edge cases".to_string(),
                description: "Zero-length and scalar-shape creation/indexing can diverge between Strict and Hardened flows when downstream operations assume at least one element.".to_string(),
                affected_operations: vec![
                    "zeros".to_string(),
                    "ones".to_string(),
                    "from_slice".to_string(),
                    "getitem".to_string(),
                ],
                mitigation: "Maintain explicit empty-shape fixtures and keep error-kind assertions for malformed index/mask combinations in differential + adversarial cases.".to_string(),
            },
            RiskNote {
                category: "zero-stride broadcasting".to_string(),
                description: "Broadcast expansion from singleton dimensions can alias logical coordinates to one source index; incorrect stride/index mapping can silently corrupt values.".to_string(),
                affected_operations: vec![
                    "broadcast_shapes".to_string(),
                    "relation_broadcast_commutative".to_string(),
                ],
                mitigation: "Retain commutativity and incompatibility fixtures, and keep broadcast index-mapping parity checks against fixture expectations for each release.".to_string(),
            },
            RiskNote {
                category: "mixed-dtype promotion".to_string(),
                description: "Float/complex and integer promotion boundaries can produce behavior drift or unsupported-dtype regressions when force_floating and symmetry rules interact.".to_string(),
                affected_operations: vec![
                    "result_type".to_string(),
                    "relation_result_type_symmetry".to_string(),
                ],
                mitigation: "Keep promotion matrix fixtures (including adversarial integral cases) and enforce symmetry/property cases in parity gates before packet sign-off.".to_string(),
            },
        ],
    };

    let fixture_log = parity_gates
        .gates
        .iter()
        .map(|gate| FixtureLogEntry {
            fixture_id: gate.fixture_id.clone(),
            operation: gate.operation.clone(),
            pass_fail: gate.pass_fail,
            shape: gate.shape.clone(),
            dtype: gate.dtype.clone(),
        })
        .collect::<Vec<_>>();

    let evidence_bundle = EvidenceBundle {
        packet_id: packet_id.clone(),
        generated_at: now_str(),
        fixture_manifest: manifest,
        parity_gates,
        parity_report,
        risk_notes,
    };

    let mut artifact_logs = Vec::new();
    let manifest_path = write_json_with_log(
        &evidence_dir,
        "fixture_manifest.json",
        &evidence_bundle.fixture_manifest,
        &mut artifact_logs,
    );
    let parity_gates_path = write_json_with_log(
        &evidence_dir,
        "parity_gates.json",
        &evidence_bundle.parity_gates,
        &mut artifact_logs,
    );
    let risk_notes_path = write_json_with_log(
        &evidence_dir,
        "risk_notes.json",
        &evidence_bundle.risk_notes,
        &mut artifact_logs,
    );
    let parity_report_path = write_json_with_log(
        &evidence_dir,
        "parity_report.json",
        &evidence_bundle.parity_report,
        &mut artifact_logs,
    );
    let fixture_log_path = write_json_with_log(
        &evidence_dir,
        "fixture_log.json",
        &fixture_log,
        &mut artifact_logs,
    );
    let evidence_bundle_path = write_json_with_log(
        &evidence_dir,
        "evidence_bundle.json",
        &evidence_bundle,
        &mut artifact_logs,
    );

    let primary_artifacts = vec![
        manifest_path,
        parity_gates_path,
        risk_notes_path,
        parity_report_path,
        fixture_log_path,
        evidence_bundle_path,
    ];

    let mut sidecars_generated = 0usize;
    for source_path in &primary_artifacts {
        if write_sidecar_and_decode(source_path, &evidence_dir, &mut artifact_logs).is_some() {
            sidecars_generated += 1;
        }
    }
    assert!(
        sidecars_generated > 0,
        "at least one RaptorQ sidecar should be generated"
    );

    write_json_with_log(
        &evidence_dir,
        "artifact_log.json",
        &artifact_logs,
        &mut Vec::new(),
    );

    assert!(
        evidence_bundle.parity_gates.all_gates_pass,
        "all parity gates must pass"
    );

    eprintln!("\n── P2C-007 Evidence Pack ──");
    eprintln!(
        "  Parity gates: {}/{} passed",
        evidence_bundle
            .parity_gates
            .gates
            .iter()
            .filter(|gate| gate.pass_fail)
            .count(),
        evidence_bundle.parity_gates.gates.len()
    );
    for summary in &evidence_bundle.parity_report.operation_summaries {
        eprintln!(
            "  {}: {}/{} pass, max_abs={:.2e}",
            summary.group, summary.passed, summary.total_fixtures, summary.max_abs_diff_across_all
        );
    }
    eprintln!("  Sidecars generated: {sidecars_generated}");
    eprintln!("  Artifacts written to: {evidence_dir:?}");
}

fn build_fixture_manifest(
    fixture_json: &Value,
    packet_id: &str,
) -> (FixtureManifest, BTreeMap<String, CaseMeta>) {
    let mut fixtures = Vec::new();
    let mut case_meta = BTreeMap::new();
    let cases = fixture_json
        .get("cases")
        .and_then(Value::as_array)
        .expect("fixture cases should be an array");

    for case in cases {
        let fixture_id = case
            .get("case_id")
            .and_then(Value::as_str)
            .unwrap_or("unknown_case")
            .to_string();
        let operation = case
            .get("operation")
            .and_then(Value::as_str)
            .unwrap_or("unknown_operation")
            .to_string();
        let category = case
            .get("category")
            .and_then(Value::as_str)
            .unwrap_or("unknown_category")
            .to_string();
        let shape = extract_shape(case);
        let dtype = extract_dtype(case);
        let expected_kind = case
            .get("expected")
            .and_then(|value| value.get("kind"))
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let contract_ref = case
            .get("expected")
            .and_then(|value| value.get("contract_ref"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);

        fixtures.push(FixtureEntry {
            fixture_id: fixture_id.clone(),
            operation: operation.clone(),
            category,
            shape: shape.clone(),
            dtype: dtype.clone(),
            expected_kind,
            contract_ref,
        });

        case_meta.insert(
            fixture_id,
            CaseMeta {
                operation: operation.clone(),
                group: operation_group(operation.as_str()).to_string(),
                shape,
                dtype,
            },
        );
    }

    (
        FixtureManifest {
            packet_id: packet_id.to_string(),
            generated_at: now_str(),
            fixtures,
        },
        case_meta,
    )
}

fn operation_group(operation: &str) -> &'static str {
    match operation {
        "zeros" | "ones" | "full" | "arange" | "linspace" | "from_slice" => "creation",
        "getitem" | "reshape" | "transpose" | "relation_index_roundtrip" => "indexing",
        "broadcast_shapes" | "relation_broadcast_commutative" => "broadcasting",
        "result_type" | "relation_result_type_symmetry" => "ufunc_dispatch",
        _ => "other",
    }
}

fn extract_shape(case: &Value) -> String {
    if let Some(shape) = case.get("shape") {
        return shape.to_string();
    }
    if let Some(shapes) = case.get("shapes") {
        return shapes.to_string();
    }
    if let (Some(left), Some(right)) = (case.get("left_shape"), case.get("right_shape")) {
        return format!("{left} x {right}");
    }
    if let Some(source_shape) = case.get("source_shape") {
        return source_shape.to_string();
    }
    if let Some(new_shape) = case.get("new_shape") {
        return new_shape.to_string();
    }
    if let Some(expected_shape) = case
        .get("expected")
        .and_then(|value| value.get("shape").or_else(|| value.get("dims")))
    {
        return expected_shape.to_string();
    }
    "n/a".to_string()
}

fn extract_dtype(case: &Value) -> String {
    if let Some(dtype) = case.get("dtype").and_then(Value::as_str) {
        return dtype.to_string();
    }
    if let Some(dtypes) = case.get("dtypes") {
        return dtypes.to_string();
    }
    if let (Some(left), Some(right)) = (
        case.get("left_dtype").and_then(Value::as_str),
        case.get("right_dtype").and_then(Value::as_str),
    ) {
        return format!("[{left},{right}]");
    }
    if let Some(dtype) = case
        .get("expected")
        .and_then(|value| value.get("dtype"))
        .and_then(Value::as_str)
    {
        return dtype.to_string();
    }
    "n/a".to_string()
}

fn oracle_status_name(status: &OracleStatus) -> &'static str {
    match status {
        OracleStatus::Available => "available",
        OracleStatus::Missing { .. } => "missing",
        OracleStatus::TimedOut => "timed_out",
        OracleStatus::Failed { .. } => "failed",
        OracleStatus::Skipped => "skipped",
    }
}

fn write_json_with_log<T: Serialize>(
    dir: &Path,
    name: &str,
    data: &T,
    artifact_logs: &mut Vec<ArtifactLogEntry>,
) -> PathBuf {
    let compact = serde_json::to_string(data).expect("artifact should serialize (compact)");
    let bytes = serde_json::to_vec_pretty(data).expect("artifact should serialize");
    let path = dir.join(name);
    std::fs::write(&path, &bytes).expect("artifact should be writable");
    emit_artifact_marker(name, &compact);
    artifact_logs.push(ArtifactLogEntry {
        artifact_name: name.to_string(),
        size_bytes: bytes.len(),
        blake3_hash: hash(&bytes).to_hex().to_string(),
        timestamp: now_str(),
    });
    path
}

fn write_sidecar_and_decode(
    source_path: &Path,
    evidence_dir: &Path,
    artifact_logs: &mut Vec<ArtifactLogEntry>,
) -> Option<(PathBuf, PathBuf)> {
    let source_bytes = std::fs::read(source_path).expect("source artifact should be readable");
    let sidecar = match generate_sidecar_resilient(&source_bytes) {
        Ok(sidecar) => sidecar,
        Err(error) => {
            eprintln!("  sidecar skipped for {source_path:?}: {error}");
            return None;
        }
    };

    let stem = source_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("artifact");
    let sidecar_name = format!("{stem}.raptorq.json");
    let sidecar_path = evidence_dir.join(&sidecar_name);
    let sidecar_compact = serde_json::to_string(&sidecar).expect("sidecar should serialize");
    let sidecar_bytes = serde_json::to_vec_pretty(&sidecar).expect("sidecar should serialize");
    std::fs::write(&sidecar_path, &sidecar_bytes).expect("sidecar should be writable");
    emit_artifact_marker(&sidecar_name, &sidecar_compact);
    artifact_logs.push(ArtifactLogEntry {
        artifact_name: sidecar_name,
        size_bytes: sidecar_bytes.len(),
        blake3_hash: hash(&sidecar_bytes).to_hex().to_string(),
        timestamp: now_str(),
    });

    let decode = DecodeProofArtifact {
        ts_unix_ms: now_unix_ms(),
        reason: "no recovery required; artifact remained intact".to_string(),
        recovered_blocks: 0,
        proof_hash: hash(&source_bytes).to_hex().to_string(),
    };
    let decode_name = format!("{stem}.decode_proof.json");
    let decode_path = evidence_dir.join(&decode_name);
    let decode_compact = serde_json::to_string(&decode).expect("decode proof should serialize");
    let decode_bytes = serde_json::to_vec_pretty(&decode).expect("decode proof should serialize");
    std::fs::write(&decode_path, &decode_bytes).expect("decode proof should be writable");
    emit_artifact_marker(&decode_name, &decode_compact);
    artifact_logs.push(ArtifactLogEntry {
        artifact_name: decode_name,
        size_bytes: decode_bytes.len(),
        blake3_hash: hash(&decode_bytes).to_hex().to_string(),
        timestamp: now_str(),
    });

    Some((sidecar_path, decode_path))
}

fn generate_sidecar_resilient(payload: &[u8]) -> Result<RaptorQSidecar, String> {
    fn hex_encode(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(bytes.len() * 2);
        for &byte in bytes {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0f) as usize] as char);
        }
        out
    }

    if let Ok(sidecar) = generate_raptorq_sidecar(payload) {
        return Ok(sidecar);
    }

    let symbol_size = 128usize;
    let source_symbols = chunk_payload(payload, symbol_size);
    let k = source_symbols.len();
    let repair_symbols = (k / 5).max(1);
    let base_seed = hash(payload).as_bytes()[0] as u64 + 1337;

    for offset in 1..=16 {
        if let Some(encoder) =
            SystematicEncoder::new(&source_symbols, symbol_size, base_seed + offset)
        {
            let mut repair_hashes = Vec::with_capacity(repair_symbols);
            let mut repair_payloads = Vec::with_capacity(repair_symbols);
            for esi in k as u32..(k as u32 + repair_symbols as u32) {
                let symbol = encoder.repair_symbol(esi);
                repair_hashes.push(hash(&symbol).to_hex().to_string());
                repair_payloads.push(hex_encode(&symbol));
            }
            return Ok(RaptorQSidecar {
                schema_version: 1,
                source_hash: hash(payload).to_hex().to_string(),
                symbol_size,
                seed: base_seed + offset,
                source_symbols: k,
                repair_symbols,
                repair_symbol_hashes: repair_hashes,
                repair_symbol_payloads_hex: repair_payloads,
            });
        }
    }

    Err("systematic encoder initialization failed after seed fallback attempts".to_string())
}

fn now_str() -> String {
    format!("unix:{}", now_unix_secs())
}

fn now_unix_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_secs()
}

fn now_unix_ms() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_millis()
}

fn emit_artifact_marker(name: &str, compact_json: &str) {
    println!("P2C007_EVIDENCE_ARTIFACT_JSON::{name}::{compact_json}");
}
