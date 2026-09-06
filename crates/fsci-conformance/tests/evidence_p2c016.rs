//! [FSCI-P2C-016-I] Final Evidence Pack for fsci-constants.
//!
//! Produces a self-contained evidence bundle at:
//!   fixtures/artifacts/P2C-016/evidence/
//!
//! Covers constant-value checks, lookup parity, and conversion helper invariants.

use blake3::hash;
use fsci_conformance::{
    HarnessConfig, RaptorQSidecar, generate_raptorq_sidecar, run_constants_packet,
    write_parity_artifacts,
};
use fsci_constants::{
    DEGREE, ELECTRON_VOLT, PI, PLANCK, SPEED_OF_LIGHT, celsius_to_kelvin, convert_temperature,
    deg2rad, ev_to_joules, find, freq_to_wavelength, joules_to_ev, kg_to_lb, lb_to_kg, rad2deg,
    value, wavelength_to_freq,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct PacketFixture {
    packet_id: String,
    family: String,
    cases: Vec<FixtureCase>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct FixtureCase {
    #[serde(alias = "case_id")]
    fixture_id: String,
    category: String,
    #[serde(alias = "function")]
    operation: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    expected_properties: Vec<String>,
}

#[derive(Debug, Serialize)]
struct FixtureManifest {
    packet_id: String,
    family: String,
    generated_at: String,
    fixtures: Vec<FixtureCase>,
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
    category: String,
    operation: String,
    pass: bool,
    max_abs_diff: f64,
    tolerance_used: f64,
    note: String,
}

#[derive(Debug, Serialize)]
struct ParityReport {
    packet_id: String,
    operation_summaries: Vec<OperationParitySummary>,
}

#[derive(Debug, Serialize)]
struct OperationParitySummary {
    category: String,
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
    parity_gates: ParityGateReport,
    parity_report: ParityReport,
    risk_notes: RiskNotesReport,
    sidecar: Option<RaptorQSidecar>,
}

fn now_str() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after epoch")
        .as_secs();
    format!("unix:{secs}")
}

fn fold_max(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0_f64, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    })
}

fn fixture_case(fixture: &PacketFixture, fixture_id: &str) -> FixtureCase {
    fixture
        .cases
        .iter()
        .find(|case| case.fixture_id == fixture_id)
        .cloned()
        .or_else(|| aggregate_fixture_case(fixture_id))
        .unwrap_or_else(|| panic!("missing fixture case {fixture_id}"))
}

fn aggregate_fixture_case(fixture_id: &str) -> Option<FixtureCase> {
    let (category, operation, description, expected_properties) = match fixture_id {
        "constants_exact_values" => (
            "fundamental_values",
            "fundamental_constants",
            "Exact or tightly specified constants match their expected CODATA-backed values.",
            &[
                "speed of light is exact",
                "Planck constant remains in the expected order of magnitude",
                "degree is pi/180",
            ][..],
        ),
        "lookup_aliases" => (
            "lookup",
            "value",
            "Named and aliased lookups resolve the expected constants while unknown names fail closed.",
            &[
                "value('speed of light') returns c",
                "value('Planck') returns h",
                "value('nonexistent') returns null",
            ][..],
        ),
        "partial_find" => (
            "lookup",
            "find",
            "Partial-name search returns stable hits for known constants.",
            &[
                "find('planck') returns at least one Planck-related constant",
                "find('charge') returns elementary charge",
            ][..],
        ),
        "temperature_roundtrip" => (
            "unit_conversion",
            "convert_temperature",
            "Temperature conversions preserve freezing/boiling references and round-trip across unit systems.",
            &[
                "0 C -> 273.15 K",
                "212 F -> 100 C",
                "C -> F -> C round-trip is stable",
            ][..],
        ),
        "energy_roundtrip" => (
            "unit_conversion",
            "ev_joules",
            "Electron-volt to joule conversion round-trips with negligible numerical drift.",
            &[
                "1 eV equals ELECTRON_VOLT joules",
                "joules_to_ev(ev_to_joules(x)) ~= x",
            ][..],
        ),
        "wave_frequency_roundtrip" => (
            "unit_conversion",
            "wave_frequency",
            "Wavelength/frequency helpers remain inverse to each other for finite positive inputs.",
            &["freq_to_wavelength(wavelength_to_freq(x)) ~= x"][..],
        ),
        "angle_and_mass_helpers" => (
            "derived_units",
            "deg_rad_mass",
            "Angle and mass helper conversions stay internally consistent.",
            &[
                "deg2rad(180) = pi",
                "rad2deg(pi/2) = 90",
                "lb_to_kg(kg_to_lb(x)) ~= x",
            ][..],
        ),
        _ => return None,
    };

    Some(FixtureCase {
        fixture_id: fixture_id.to_owned(),
        category: category.to_owned(),
        operation: operation.to_owned(),
        description: description.to_owned(),
        expected_properties: expected_properties
            .iter()
            .map(|property| (*property).to_owned())
            .collect(),
    })
}

fn gate_from_case(
    fixture: &PacketFixture,
    fixture_id: &str,
    pass: bool,
    max_abs_diff: f64,
    tolerance_used: f64,
    note: impl Into<String>,
) -> ParityGate {
    let case = fixture_case(fixture, fixture_id);
    ParityGate {
        fixture_id: case.fixture_id.clone(),
        category: case.category.clone(),
        operation: case.operation.clone(),
        pass,
        max_abs_diff,
        tolerance_used,
        note: note.into(),
    }
}

fn check_exact_values(fixture: &PacketFixture) -> ParityGate {
    let speed_diff = (SPEED_OF_LIGHT - 299_792_458.0).abs();
    let planck_diff = (PLANCK - 6.626_070_15e-34).abs();
    let degree_diff = (DEGREE - PI / 180.0).abs();
    let max_abs_diff = fold_max([speed_diff, planck_diff, degree_diff].into_iter());
    gate_from_case(
        fixture,
        "constants_exact_values",
        speed_diff == 0.0 && planck_diff < 1e-45 && degree_diff < 1e-16,
        max_abs_diff,
        1e-16,
        format!(
            "speed_diff={speed_diff:.3e}, planck_diff={planck_diff:.3e}, degree_diff={degree_diff:.3e}"
        ),
    )
}

fn check_lookup_aliases(fixture: &PacketFixture) -> ParityGate {
    let checks = [
        value("speed of light") == Some(SPEED_OF_LIGHT),
        value("Planck") == Some(PLANCK),
        value("sigma").is_some(),
        value("nonexistent").is_none(),
    ];
    let pass = checks.into_iter().all(std::convert::identity);
    gate_from_case(
        fixture,
        "lookup_aliases",
        pass,
        if pass { 0.0 } else { 1.0 },
        0.0,
        "lookup aliases and unknown-name behavior verified",
    )
}

fn check_partial_find(fixture: &PacketFixture) -> ParityGate {
    let planck_hits = find("planck");
    let charge_hits = find("charge");
    let pass = planck_hits
        .iter()
        .any(|(name, _)| name.to_lowercase().contains("planck"))
        && charge_hits
            .iter()
            .any(|(name, _)| name.to_lowercase().contains("charge"));
    gate_from_case(
        fixture,
        "partial_find",
        pass,
        if pass { 0.0 } else { 1.0 },
        0.0,
        format!(
            "planck_hits={}, charge_hits={}",
            planck_hits.len(),
            charge_hits.len()
        ),
    )
}

fn check_temperature_roundtrip(fixture: &PacketFixture) -> ParityGate {
    let freezing_diff = (celsius_to_kelvin(0.0) - 273.15).abs();
    let boiling_diff = (convert_temperature(212.0, "fahrenheit", "celsius").unwrap() - 100.0).abs();
    let roundtrip = convert_temperature(
        convert_temperature(37.5, "celsius", "fahrenheit").unwrap(),
        "fahrenheit",
        "celsius",
    )
    .unwrap();
    let roundtrip_diff = (roundtrip - 37.5).abs();
    let max_abs_diff = fold_max([freezing_diff, boiling_diff, roundtrip_diff].into_iter());
    gate_from_case(
        fixture,
        "temperature_roundtrip",
        freezing_diff < 1e-12 && boiling_diff < 1e-12 && roundtrip_diff < 1e-10,
        max_abs_diff,
        1e-10,
        format!(
            "freeze={freezing_diff:.3e}, boil={boiling_diff:.3e}, roundtrip={roundtrip_diff:.3e}"
        ),
    )
}

fn check_energy_roundtrip(fixture: &PacketFixture) -> ParityGate {
    let joules = ev_to_joules(7.5);
    let exact_diff = (ev_to_joules(1.0) - ELECTRON_VOLT).abs();
    let roundtrip_diff = (joules_to_ev(joules) - 7.5).abs();
    let max_abs_diff = fold_max([exact_diff, roundtrip_diff].into_iter());
    gate_from_case(
        fixture,
        "energy_roundtrip",
        exact_diff < 1e-30 && roundtrip_diff < 1e-12,
        max_abs_diff,
        1e-12,
        format!("exact={exact_diff:.3e}, roundtrip={roundtrip_diff:.3e}"),
    )
}

fn check_wave_frequency_roundtrip(fixture: &PacketFixture) -> ParityGate {
    let wavelength = 632.8e-9;
    let recovered = freq_to_wavelength(wavelength_to_freq(wavelength));
    let diff = (recovered - wavelength).abs();
    gate_from_case(
        fixture,
        "wave_frequency_roundtrip",
        diff < 1e-18,
        diff,
        1e-18,
        format!("recovered wavelength diff={diff:.3e}"),
    )
}

fn check_angle_and_mass_helpers(fixture: &PacketFixture) -> ParityGate {
    let deg_diff = (deg2rad(180.0) - PI).abs();
    let rad_diff = (rad2deg(PI / 2.0) - 90.0).abs();
    let mass_diff = (lb_to_kg(kg_to_lb(12.5)) - 12.5).abs();
    let max_abs_diff = fold_max([deg_diff, rad_diff, mass_diff].into_iter());
    gate_from_case(
        fixture,
        "angle_and_mass_helpers",
        deg_diff < 1e-12 && rad_diff < 1e-12 && mass_diff < 1e-12,
        max_abs_diff,
        1e-12,
        format!("deg_diff={deg_diff:.3e}, rad_diff={rad_diff:.3e}, mass_diff={mass_diff:.3e}"),
    )
}

#[test]
fn evidence_p2c016_final_pack() {
    let fixture_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/FSCI-P2C-016_constants_core.json");
    let evidence_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/P2C-016/evidence");
    std::fs::create_dir_all(&evidence_dir).expect("evidence directory should be creatable");

    let fixture_raw =
        std::fs::read_to_string(&fixture_path).expect("P2C-016 fixture file should be readable");
    let fixture: PacketFixture =
        serde_json::from_str(&fixture_raw).expect("P2C-016 fixture file should parse");

    let manifest = FixtureManifest {
        packet_id: fixture.packet_id.clone(),
        family: fixture.family.clone(),
        generated_at: now_str(),
        fixtures: fixture.cases.clone(),
    };

    let gates = vec![
        check_exact_values(&fixture),
        check_lookup_aliases(&fixture),
        check_partial_find(&fixture),
        check_temperature_roundtrip(&fixture),
        check_energy_roundtrip(&fixture),
        check_wave_frequency_roundtrip(&fixture),
        check_angle_and_mass_helpers(&fixture),
    ];

    assert!(
        fixture.cases.len() >= gates.len(),
        "constants fixture should retain detailed cases alongside aggregate parity gates"
    );

    let all_gates_pass = gates.iter().all(|gate| gate.pass);
    let parity_gates = ParityGateReport {
        packet_id: fixture.packet_id.clone(),
        all_gates_pass,
        gates,
    };

    let mut grouped: BTreeMap<String, Vec<&ParityGate>> = BTreeMap::new();
    for gate in &parity_gates.gates {
        grouped.entry(gate.category.clone()).or_default().push(gate);
    }
    let operation_summaries = grouped
        .into_iter()
        .map(|(category, gates)| OperationParitySummary {
            category,
            total_fixtures: gates.len(),
            passed: gates.iter().filter(|gate| gate.pass).count(),
            failed: gates.iter().filter(|gate| !gate.pass).count(),
            max_abs_diff_across_all: fold_max(gates.iter().map(|gate| gate.max_abs_diff)),
        })
        .collect::<Vec<_>>();

    let parity_report = ParityReport {
        packet_id: fixture.packet_id.clone(),
        operation_summaries,
    };

    let risk_notes = RiskNotesReport {
        packet_id: fixture.packet_id.clone(),
        notes: vec![
            RiskNote {
                category: "alias_lookup_surface".to_string(),
                description: "Short aliases like `c`, `h`, `sigma`, and `ev` are observable API surface; accidental renames or case-handling changes would silently break callers.".to_string(),
                affected_operations: vec!["value".to_string(), "find".to_string()],
                mitigation: "Keep alias coverage in conformance fixtures and fail closed (`None`) for unknown names instead of returning approximate matches.".to_string(),
            },
            RiskNote {
                category: "unit_roundtrip_drift".to_string(),
                description: "Derived helpers compose multiple floating-point operations, so small arithmetic regressions can accumulate in round-trip conversions.".to_string(),
                affected_operations: vec![
                    "convert_temperature".to_string(),
                    "ev_to_joules".to_string(),
                    "joules_to_ev".to_string(),
                    "wavelength_to_freq".to_string(),
                    "freq_to_wavelength".to_string(),
                ],
                mitigation: "Preserve explicit round-trip fixtures with tight tolerances and keep exact-value checks for constants that should not drift.".to_string(),
            },
            RiskNote {
                category: "mixed_exact_and_approximate_constants".to_string(),
                description: "Some constants are exact by definition while others are represented to a practical precision. Treating both classes identically can either over-tighten or under-specify parity gates.".to_string(),
                affected_operations: vec![
                    "fundamental_constants".to_string(),
                    "deg2rad".to_string(),
                    "rad2deg".to_string(),
                ],
                mitigation: "Use exact equality for definition-fixed constants and explicit tolerances for derived/approximate constants in the evidence pack.".to_string(),
            },
        ],
    };

    let evidence = EvidenceBundle {
        packet_id: fixture.packet_id.clone(),
        generated_at: now_str(),
        fixture_manifest: manifest,
        parity_gates,
        parity_report,
        risk_notes,
        sidecar: None,
    };

    let bundle_json = serde_json::to_string_pretty(&evidence).expect("bundle should serialize");
    std::fs::write(evidence_dir.join("evidence_bundle.json"), &bundle_json)
        .expect("bundle should be written");

    let sidecar = generate_raptorq_sidecar(bundle_json.as_bytes()).expect("sidecar should build");
    std::fs::write(
        evidence_dir.join("evidence_bundle.raptorq.json"),
        serde_json::to_string_pretty(&sidecar).expect("sidecar should serialize"),
    )
    .expect("sidecar should be written");

    std::fs::write(
        evidence_dir.join("fixture_manifest.json"),
        serde_json::to_string_pretty(&evidence.fixture_manifest)
            .expect("manifest should serialize"),
    )
    .expect("manifest should be written");
    std::fs::write(
        evidence_dir.join("parity_gates.json"),
        serde_json::to_string_pretty(&evidence.parity_gates).expect("gates should serialize"),
    )
    .expect("gates should be written");
    std::fs::write(
        evidence_dir.join("parity_report.json"),
        serde_json::to_string_pretty(&evidence.parity_report).expect("report should serialize"),
    )
    .expect("report should be written");
    std::fs::write(
        evidence_dir.join("risk_notes.json"),
        serde_json::to_string_pretty(&evidence.risk_notes).expect("notes should serialize"),
    )
    .expect("notes should be written");
    std::fs::write(
        evidence_dir.join("evidence_bundle.blake3"),
        hash(bundle_json.as_bytes()).to_hex().to_string(),
    )
    .expect("bundle hash should be written");

    for gate in &evidence.parity_gates.gates {
        let status = if gate.pass { "PASS" } else { "FAIL" };
        eprintln!(
            "  {status}: {} {} — {}",
            gate.operation, gate.fixture_id, gate.note
        );
    }
    assert!(all_gates_pass, "all P2C-016 parity gates must pass");
}

#[test]
fn evidence_p2c016_root_parity_report() {
    // frankenscipy-8oub0: 16 of 18 FSCI packets carry a root
    // parity_report.json; FSCI-P2C-016 only had a nested
    // differential/parity_report.json. Generate the root triple through the
    // constants dispatch lane — run_constants_packet -> write_parity_artifacts,
    // the same path evidence_p2c012 uses for FSCI-P2C-012 — so the packet
    // topology converges on the flat layout. The writer is byte-stable
    // modulo generated_unix_ms (okuhi), so committed artifacts do not churn.
    let config = HarnessConfig::default_paths();
    let packet_report = run_constants_packet(&config, "FSCI-P2C-016_constants_core.json")
        .expect("constants dispatch packet must run");
    let artifacts = write_parity_artifacts(&config, &packet_report)
        .expect("root parity artifacts must be written");
    assert!(
        artifacts.report_path.exists(),
        "{:?}",
        artifacts.report_path
    );
    assert!(
        artifacts.sidecar_path.exists(),
        "{:?}",
        artifacts.sidecar_path
    );
    assert!(
        artifacts.decode_proof_path.exists(),
        "{:?}",
        artifacts.decode_proof_path
    );
    assert_eq!(
        artifacts.report_path.parent().unwrap().file_name().unwrap(),
        "FSCI-P2C-016",
        "report must land at the packet root, not a subdirectory"
    );
}
