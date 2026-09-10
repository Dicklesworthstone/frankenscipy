#![forbid(unsafe_code)]

use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_dir(suffix: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    std::env::temp_dir().join(format!(
        "fsci_conformance_e2e_{suffix}_{}_{}",
        std::process::id(),
        nonce
    ))
}

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|error| {
            panic!("failed to create {}: {error}", parent.display());
        });
    }
    fs::write(path, contents).unwrap_or_else(|error| {
        panic!("failed to write {}: {error}", path.display());
    });
}

fn copy_file(src: &Path, dst: &Path) {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|error| {
            panic!("failed to create {}: {error}", parent.display());
        });
    }
    fs::copy(src, dst).unwrap_or_else(|error| {
        panic!(
            "failed to copy {} -> {}: {error}",
            src.display(),
            dst.display()
        );
    });
}

fn collect_named_files(root: &Path, file_name: &str, out: &mut Vec<PathBuf>) {
    if !root.exists() {
        return;
    }
    let entries = fs::read_dir(root).unwrap_or_else(|error| {
        panic!("failed to read {}: {error}", root.display());
    });
    for entry in entries {
        let entry = entry.unwrap_or_else(|error| {
            panic!("failed to read entry in {}: {error}", root.display());
        });
        let path = entry.path();
        if path.is_dir() {
            collect_named_files(&path, file_name, out);
        } else if path.file_name().and_then(|name| name.to_str()) == Some(file_name) {
            out.push(path);
        }
    }
}

#[test]
fn e2e_orchestrator_runs_arrayapi_packet_with_minimum_scenarios() {
    let temp_root = unique_temp_dir("arrayapi");
    let artifact_root = temp_root.join("artifacts");
    let packet_dir = artifact_root.join("FSCI-P2C-007");
    let fixture_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let source_packet_dir = fixture_root.join("artifacts/FSCI-P2C-007");
    let source_scenarios_dir = source_packet_dir.join("e2e/scenarios");
    let target_scenarios_dir = packet_dir.join("e2e/scenarios");

    assert!(
        source_scenarios_dir.exists(),
        "expected source scenarios at {}",
        source_scenarios_dir.display()
    );

    copy_file(
        &source_packet_dir.join("anchor/behavior_ledger.json"),
        &packet_dir.join("anchor/behavior_ledger.json"),
    );
    copy_file(
        &source_packet_dir.join("contracts/contract_table.json"),
        &packet_dir.join("contracts/contract_table.json"),
    );
    copy_file(
        &source_packet_dir.join("threats/threat_matrix.json"),
        &packet_dir.join("threats/threat_matrix.json"),
    );

    let scenario_entries = fs::read_dir(&source_scenarios_dir).unwrap_or_else(|error| {
        panic!(
            "failed to read source scenarios {}: {error}",
            source_scenarios_dir.display()
        );
    });
    let mut copied_scenarios = 0_usize;
    for entry in scenario_entries {
        let entry = entry.unwrap_or_else(|error| {
            panic!(
                "failed to read source scenario entry in {}: {error}",
                source_scenarios_dir.display()
            );
        });
        let source = entry.path();
        let is_json = source
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("json"));
        if !is_json {
            continue;
        }
        let file_name = source
            .file_name()
            .expect("scenario file should have file name");
        let target = target_scenarios_dir.join(file_name);
        copy_file(&source, &target);
        copied_scenarios += 1;
    }

    assert!(
        copied_scenarios >= 8,
        "expected at least 8 copied scenarios, got {copied_scenarios}"
    );

    let output = Command::new(env!("CARGO_BIN_EXE_e2e_orchestrator"))
        .arg("--artifact-root")
        .arg(&artifact_root)
        .arg("--fixture-root")
        .arg(&fixture_root)
        .arg("--packet")
        .arg("FSCI-P2C-007")
        .output()
        .expect("failed to execute e2e_orchestrator");

    assert!(
        output.status.success(),
        "expected success for arrayapi packet run; stdout={}; stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let mut summary_files = Vec::new();
    collect_named_files(
        &packet_dir.join("e2e/runs"),
        "summary.json",
        &mut summary_files,
    );
    assert!(
        summary_files.len() >= 8,
        "expected >= 8 summary bundles, got {}",
        summary_files.len()
    );

    let mut event_files = Vec::new();
    collect_named_files(
        &packet_dir.join("e2e/runs"),
        "events.jsonl",
        &mut event_files,
    );
    assert_eq!(
        event_files.len(),
        summary_files.len(),
        "events/summary bundle count mismatch"
    );

    for summary_file in &summary_files {
        let raw = fs::read_to_string(summary_file).unwrap_or_else(|error| {
            panic!("failed to read {}: {error}", summary_file.display());
        });
        let json: Value = serde_json::from_str(&raw).unwrap_or_else(|error| {
            panic!("failed to parse {}: {error}", summary_file.display());
        });
        assert_eq!(json["packet_id"], "FSCI-P2C-007");
        assert_eq!(json["passed"], true);
        assert!(json["replay_command"].is_string());
    }

    let events_raw = fs::read_to_string(&event_files[0]).expect("failed to read events log");
    let first: Value = serde_json::from_str(
        events_raw
            .lines()
            .next()
            .expect("missing first events line"),
    )
    .expect("failed to parse first events line");
    assert!(first["scenario_id"].is_string());
    assert_eq!(first["step_name"], "setup");
    assert!(first["environment"]["seed"].is_number());
    assert!(first["environment"]["input_hashes"].is_array());
}

#[test]
fn e2e_orchestrator_runs_direct_e2e_json_layout_and_ignores_runs() {
    let temp_root = unique_temp_dir("direct_layout");
    let artifact_root = temp_root.join("artifacts");
    let packet_dir = artifact_root.join("FSCI-P2C-008");
    let fixture_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let target_e2e_dir = packet_dir.join("e2e");
    let target_runs_dir = target_e2e_dir.join("runs/existing-run/ignored");
    let sentinel_path = packet_dir.join("sentinel.txt");

    write_file(&sentinel_path, "sentinel");
    write_file(
        &target_e2e_dir.join("direct_layout_scenario.json"),
        r#"{
  "scenario_id": "direct_layout_scenario",
  "seed": 8008,
  "steps": [
    {
      "step_name": "ensure_sentinel_present",
      "kind": "check_path_exists",
      "path": "FSCI-P2C-008/sentinel.txt"
    },
    {
      "step_name": "run_validate_tol_fixture",
      "kind": "run_differential_fixture",
      "fixture": "FSCI-P2C-001_validate_tol.json"
    }
  ]
}"#,
    );

    write_file(
        &target_runs_dir.join("summary.json"),
        r#"{"scenario_id":"should_not_be_discovered","passed":true}"#,
    );

    let output = Command::new(env!("CARGO_BIN_EXE_e2e_orchestrator"))
        .arg("--artifact-root")
        .arg(&artifact_root)
        .arg("--fixture-root")
        .arg(&fixture_root)
        .arg("--packet")
        .arg("FSCI-P2C-008")
        .output()
        .expect("failed to execute e2e_orchestrator");

    assert!(
        output.status.success(),
        "expected success for direct-layout packet run; stdout={}; stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let mut summary_files = Vec::new();
    collect_named_files(
        &packet_dir.join("e2e/runs"),
        "summary.json",
        &mut summary_files,
    );
    assert_eq!(
        summary_files.len(),
        2,
        "expected one new scenario bundle plus the pre-existing ignored run artifact"
    );

    let scenario_summaries = summary_files
        .iter()
        .filter(|path| {
            !path
                .components()
                .any(|component| component.as_os_str() == "existing-run")
        })
        .count();
    assert_eq!(
        scenario_summaries, 1,
        "orchestrator should only execute the direct-layout scenario file"
    );
}

#[test]
fn e2e_orchestrator_runs_scenario_and_emits_forensic_bundle() {
    let temp_root = unique_temp_dir("pass");
    let artifact_root = temp_root.join("artifacts");
    let packet_dir = artifact_root.join("FSCI-P2C-TEST");
    let scenario_path = packet_dir.join("e2e/scenarios/scenario_pass.json");
    let sentinel_path = packet_dir.join("sentinel.txt");
    write_file(&sentinel_path, "sentinel");
    write_file(
        &scenario_path,
        r#"{
  "scenario_id": "scenario_pass",
  "seed": 123,
  "steps": [
    {
      "step_name": "ensure_sentinel_present",
      "kind": "check_path_exists",
      "path": "FSCI-P2C-TEST/sentinel.txt"
    },
    {
      "step_name": "run_validate_tol_fixture",
      "kind": "run_differential_fixture",
      "fixture": "FSCI-P2C-001_validate_tol.json"
    }
  ]
}"#,
    );

    let fixture_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let output = Command::new(env!("CARGO_BIN_EXE_e2e_orchestrator"))
        .arg("--artifact-root")
        .arg(&artifact_root)
        .arg("--fixture-root")
        .arg(&fixture_root)
        .output()
        .expect("failed to execute e2e_orchestrator");

    assert!(
        output.status.success(),
        "expected success; stdout={}; stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let mut event_files = Vec::new();
    collect_named_files(
        &packet_dir.join("e2e/runs"),
        "events.jsonl",
        &mut event_files,
    );
    assert_eq!(
        event_files.len(),
        1,
        "expected exactly one events.jsonl file"
    );

    let events_raw = fs::read_to_string(&event_files[0]).expect("failed to read events log");
    let lines: Vec<&str> = events_raw.lines().collect();
    assert!(
        lines.len() >= 6,
        "expected lifecycle + scenario entries, got {}",
        lines.len()
    );

    let first: Value = serde_json::from_str(lines.first().expect("missing first log line"))
        .expect("failed to parse first log line as json");
    assert_eq!(first["scenario_id"], "scenario_pass");
    assert_eq!(first["step_name"], "setup");
    assert!(first["environment"]["rust_version"].is_string());
    assert_eq!(first["environment"]["seed"], 123);
    assert!(first["environment"]["input_hashes"].is_array());

    assert!(
        lines
            .iter()
            .any(|line| line.contains("ensure_sentinel_present")),
        "missing check_path_exists step log"
    );
    assert!(
        lines
            .iter()
            .any(|line| line.contains("run_validate_tol_fixture")),
        "missing run_differential_fixture step log"
    );

    let mut summary_files = Vec::new();
    collect_named_files(
        &packet_dir.join("e2e/runs"),
        "summary.json",
        &mut summary_files,
    );
    assert_eq!(
        summary_files.len(),
        1,
        "expected exactly one summary.json file"
    );
    let summary_raw = fs::read_to_string(&summary_files[0]).expect("failed to read summary file");
    let summary_json: Value =
        serde_json::from_str(&summary_raw).expect("failed to parse summary file");
    assert_eq!(summary_json["scenario_id"], "scenario_pass");
    assert_eq!(summary_json["passed"], true);
}

#[test]
fn e2e_orchestrator_emits_replay_command_as_last_log_line_on_failure() {
    let temp_root = unique_temp_dir("fail");
    let artifact_root = temp_root.join("artifacts");
    let packet_dir = artifact_root.join("FSCI-P2C-FAIL");
    let scenario_path = packet_dir.join("e2e/scenarios/scenario_fail.json");
    write_file(
        &scenario_path,
        r#"{
  "scenario_id": "scenario_fail",
  "seed": 7,
  "steps": [
    {
      "step_name": "expect_missing_path",
      "kind": "check_path_exists",
      "path": "FSCI-P2C-FAIL/does-not-exist.txt"
    }
  ]
}"#,
    );

    let output = Command::new(env!("CARGO_BIN_EXE_e2e_orchestrator"))
        .arg("--artifact-root")
        .arg(&artifact_root)
        .arg("--scenario")
        .arg("scenario_fail")
        .output()
        .expect("failed to execute e2e_orchestrator");

    assert!(
        !output.status.success(),
        "expected failure exit code for failing scenario"
    );
    assert_eq!(
        output.status.code(),
        Some(1),
        "failing scenarios should return exit code 1"
    );

    let mut event_files = Vec::new();
    collect_named_files(
        &packet_dir.join("e2e/runs"),
        "events.jsonl",
        &mut event_files,
    );
    assert_eq!(
        event_files.len(),
        1,
        "expected exactly one events.jsonl file"
    );

    let events_raw = fs::read_to_string(&event_files[0]).expect("failed to read events log");
    let lines: Vec<&str> = events_raw.lines().collect();
    let last: Value = serde_json::from_str(lines.last().expect("missing replay line"))
        .expect("failed to parse last log line");

    assert_eq!(last["step_name"], "replay_command");
    assert_eq!(last["scenario_id"], "scenario_fail");
    let replay = last["replay_command"]
        .as_str()
        .expect("replay_command should be present");
    assert!(
        replay.contains("--scenario scenario_fail"),
        "replay command should include scenario filter, got: {replay}"
    );
}
