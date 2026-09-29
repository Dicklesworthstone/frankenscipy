#![forbid(unsafe_code)]
//! E2E differential fixture tests for FSCI-P2C-016 (constants).
//!
//! The constants packet already has a SciPy oracle capture script and a
//! fixture. This harness closes the missing executable E2E lane by dispatching
//! the fixture cases against `fsci-constants`, checking the same scalar/error
//! contract, and writing topology-compatible forensic artifacts.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_constants::{
    ATMOSPHERE, AVOGADRO, BOLTZMANN, CALORIE, DEGREE, ELEMENTARY_CHARGE, HBAR, INCH, PLANCK, POUND,
    SPEED_OF_LIGHT, convert_temperature, deg2rad, ev_to_joules, freq_to_wavelength, joules_to_ev,
    kg_to_lb, lb_to_kg, rad2deg, wavelength_to_freq,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const PACKET_ID: &str = "FSCI-P2C-016";
const FIXTURE_JSON: &str = include_str!("../fixtures/FSCI-P2C-016_constants_core.json");

#[derive(Debug, Deserialize)]
struct PacketFixture {
    packet_id: String,
    family: String,
    cases: Vec<FixtureCase>,
}

#[derive(Debug, Deserialize)]
struct FixtureCase {
    case_id: String,
    category: String,
    mode: String,
    function: String,
    args: Vec<Value>,
    expected: Expected,
}

#[derive(Debug, Deserialize)]
struct Expected {
    kind: String,
    value: Option<f64>,
    atol: Option<f64>,
    rtol: Option<f64>,
    error: Option<String>,
    contract_ref: String,
}

#[derive(Debug)]
enum Actual {
    Scalar(f64),
    Error(String),
}

#[derive(Debug, Serialize)]
struct EventEntry {
    packet_id: String,
    family: String,
    case_id: String,
    category: String,
    function: String,
    mode: String,
    expected_kind: String,
    outcome: String,
    max_abs_diff: Option<f64>,
    tolerance: Option<f64>,
    message: String,
    duration_ns: u128,
}

#[derive(Debug, Serialize)]
struct RunSummary {
    packet_id: String,
    family: String,
    run_id: String,
    case_count: usize,
    passed: usize,
    failed: usize,
    max_abs_diff: f64,
    generated_unix_ms: u128,
    replay_command: String,
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn run_id() -> String {
    format!("run-{}", now_unix_ms())
}

fn output_dir(run_id: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/artifacts")
        .join(PACKET_ID)
        .join("e2e/runs")
        .join(run_id)
}

fn replay_command() -> String {
    "rch exec -- env CARGO_TARGET_DIR=/tmp/rch_target_frankenscipy_codex_cod_frankenscipy_1e9v cargo test -p fsci-conformance --test e2e_constants -- --nocapture".to_string()
}

fn constant_value(name: &str) -> Option<f64> {
    match name.to_ascii_uppercase().as_str() {
        "SPEED_OF_LIGHT" => Some(SPEED_OF_LIGHT),
        "PLANCK" => Some(PLANCK),
        "HBAR" => Some(HBAR),
        "ELEMENTARY_CHARGE" => Some(ELEMENTARY_CHARGE),
        "BOLTZMANN" => Some(BOLTZMANN),
        "AVOGADRO" => Some(AVOGADRO),
        "DEGREE" => Some(DEGREE),
        "ATMOSPHERE" => Some(ATMOSPHERE),
        "CALORIE" => Some(CALORIE),
        "POUND" => Some(POUND),
        "INCH" => Some(INCH),
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

fn string_arg(case: &FixtureCase, index: usize) -> Result<&str, String> {
    case.args
        .get(index)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{} arg {index} must be a string", case.case_id))
}

fn f64_arg(case: &FixtureCase, index: usize) -> Result<f64, String> {
    case.args
        .get(index)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{} arg {index} must be numeric", case.case_id))
}

fn run_case(case: &FixtureCase) -> Actual {
    let result = match case.function.as_str() {
        "constant_value" => string_arg(case, 0).and_then(|name| {
            constant_value(name).ok_or_else(|| format!("unknown constant: {name}"))
        }),
        "convert_temperature" => {
            let value = f64_arg(case, 0);
            let from = string_arg(case, 1);
            let to = string_arg(case, 2);
            match (value, from, to) {
                (Ok(value), Ok(from), Ok(to)) => convert_temperature(value, from, to),
                (Err(err), _, _) | (_, Err(err), _) | (_, _, Err(err)) => Err(err),
            }
        }
        "ev_to_joules" => f64_arg(case, 0).map(ev_to_joules),
        "joules_to_ev" => f64_arg(case, 0).map(joules_to_ev),
        "wavelength_to_freq" => f64_arg(case, 0).map(wavelength_to_freq),
        "freq_to_wavelength" => f64_arg(case, 0).map(freq_to_wavelength),
        "deg2rad" => f64_arg(case, 0).map(deg2rad),
        "rad2deg" => f64_arg(case, 0).map(rad2deg),
        "lb_to_kg" => f64_arg(case, 0).map(lb_to_kg),
        "kg_to_lb" => f64_arg(case, 0).map(kg_to_lb),
        other => Err(format!("unsupported function: {other}")),
    };

    match result {
        Ok(value) => Actual::Scalar(value),
        Err(err) => Actual::Error(err),
    }
}

fn tolerance(expected: &Expected) -> f64 {
    expected
        .atol
        .unwrap_or(0.0)
        .max(expected.rtol.unwrap_or(0.0) * expected.value.unwrap_or_default().abs())
}

fn evaluate_case(packet: &PacketFixture, case: &FixtureCase) -> EventEntry {
    let start = Instant::now();
    let actual = run_case(case);
    let elapsed = start.elapsed().as_nanos();

    match (&case.expected.kind[..], actual) {
        ("scalar", Actual::Scalar(value)) => {
            let tol = tolerance(&case.expected);
            // JSON has no NaN literal, so every scalar expectation in the fixture is a number,
            // and a scalar case without a `value` is a fixture defect. It fails here; it used to
            // default to NaN, and a NaN-equals-NaN clause then passed it against a NaN result.
            let (pass, diff, message) = match case.expected.value {
                Some(expected) => {
                    let diff = (value - expected).abs();
                    (
                        diff <= tol,
                        Some(diff),
                        format!(
                            "actual={value:.17e}, expected={expected:.17e}, diff={diff:.3e}, contract={}",
                            case.expected.contract_ref
                        ),
                    )
                }
                None => (
                    false,
                    None,
                    format!(
                        "scalar case has no expected value in the fixture; actual={value:.17e}, contract={}",
                        case.expected.contract_ref
                    ),
                ),
            };
            EventEntry {
                packet_id: packet.packet_id.clone(),
                family: packet.family.clone(),
                case_id: case.case_id.clone(),
                category: case.category.clone(),
                function: case.function.clone(),
                mode: case.mode.clone(),
                expected_kind: case.expected.kind.clone(),
                outcome: if pass { "PASS" } else { "FAIL" }.to_string(),
                max_abs_diff: diff,
                tolerance: Some(tol),
                message,
                duration_ns: elapsed,
            }
        }
        ("error", Actual::Error(err)) => {
            // `contains("")` matches every error, so a case without an expected error text fails
            // instead of passing whatever error came back.
            let expected = case.expected.error.as_deref().unwrap_or_default();
            let pass = !expected.is_empty() && err.contains(expected);
            EventEntry {
                packet_id: packet.packet_id.clone(),
                family: packet.family.clone(),
                case_id: case.case_id.clone(),
                category: case.category.clone(),
                function: case.function.clone(),
                mode: case.mode.clone(),
                expected_kind: case.expected.kind.clone(),
                outcome: if pass { "PASS" } else { "FAIL" }.to_string(),
                max_abs_diff: None,
                tolerance: None,
                message: format!("actual_error={err:?}, expected_contains={expected:?}"),
                duration_ns: elapsed,
            }
        }
        ("scalar", Actual::Error(err)) => EventEntry {
            packet_id: packet.packet_id.clone(),
            family: packet.family.clone(),
            case_id: case.case_id.clone(),
            category: case.category.clone(),
            function: case.function.clone(),
            mode: case.mode.clone(),
            expected_kind: case.expected.kind.clone(),
            outcome: "FAIL".to_string(),
            max_abs_diff: None,
            tolerance: Some(tolerance(&case.expected)),
            message: format!("expected scalar but got error: {err}"),
            duration_ns: elapsed,
        },
        ("error", Actual::Scalar(value)) => EventEntry {
            packet_id: packet.packet_id.clone(),
            family: packet.family.clone(),
            case_id: case.case_id.clone(),
            category: case.category.clone(),
            function: case.function.clone(),
            mode: case.mode.clone(),
            expected_kind: case.expected.kind.clone(),
            outcome: "FAIL".to_string(),
            max_abs_diff: None,
            tolerance: None,
            message: format!("expected error but got scalar: {value:.17e}"),
            duration_ns: elapsed,
        },
        (kind, actual) => EventEntry {
            packet_id: packet.packet_id.clone(),
            family: packet.family.clone(),
            case_id: case.case_id.clone(),
            category: case.category.clone(),
            function: case.function.clone(),
            mode: case.mode.clone(),
            expected_kind: case.expected.kind.clone(),
            outcome: "FAIL".to_string(),
            max_abs_diff: None,
            tolerance: None,
            message: format!("unsupported expected kind {kind:?} for actual {actual:?}"),
            duration_ns: elapsed,
        },
    }
}

fn write_artifacts(packet: &PacketFixture, events: &[EventEntry]) -> Result<(), String> {
    let rid = run_id();
    let dir = output_dir(&rid);
    fs::create_dir_all(&dir).map_err(|err| {
        format!(
            "failed to create constants e2e dir {}: {err}",
            dir.display()
        )
    })?;

    let events_path = dir.join("events.jsonl");
    let file = fs::File::create(&events_path)
        .map_err(|err| format!("failed to create {}: {err}", events_path.display()))?;
    let mut writer = BufWriter::new(file);
    for event in events {
        serde_json::to_writer(&mut writer, event)
            .map_err(|err| format!("failed to serialize constants event: {err}"))?;
        writer
            .write_all(b"\n")
            .map_err(|err| format!("failed to write constants event: {err}"))?;
    }
    writer
        .flush()
        .map_err(|err| format!("failed to flush {}: {err}", events_path.display()))?;

    let passed = events
        .iter()
        .filter(|event| event.outcome == "PASS")
        .count();
    let max_abs_diff = events
        .iter()
        .filter_map(|event| event.max_abs_diff)
        .fold(0.0_f64, f64::max);
    let summary = RunSummary {
        packet_id: packet.packet_id.clone(),
        family: packet.family.clone(),
        run_id: rid,
        case_count: events.len(),
        passed,
        failed: events.len() - passed,
        max_abs_diff,
        generated_unix_ms: now_unix_ms(),
        replay_command: replay_command(),
    };
    let summary_path = dir.join("summary.json");
    let summary_json = serde_json::to_vec_pretty(&summary)
        .map_err(|err| format!("failed to serialize constants summary: {err}"))?;
    fs::write(&summary_path, summary_json)
        .map_err(|err| format!("failed to write {}: {err}", summary_path.display()))?;

    Ok(())
}

#[test]
fn e2e_constants_fixture_matches_expected_values() {
    let packet: PacketFixture =
        serde_json::from_str(FIXTURE_JSON).expect("FSCI-P2C-016 fixture should parse");
    assert_eq!(packet.packet_id, PACKET_ID);
    assert_eq!(packet.family, "constants_core");
    assert!(
        !packet.cases.is_empty(),
        "constants fixture must not be empty"
    );

    let events = packet
        .cases
        .iter()
        .map(|case| evaluate_case(&packet, case))
        .collect::<Vec<_>>();

    write_artifacts(&packet, &events).expect("constants e2e artifacts should be written");

    let failures = events
        .iter()
        .filter(|event| event.outcome != "PASS")
        .map(|event| format!("{}: {}", event.case_id, event.message))
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty(),
        "constants fixture mismatches:\n{}",
        failures.join("\n")
    );
}
