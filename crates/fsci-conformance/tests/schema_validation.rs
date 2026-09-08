//! Schema validation integration test for bd-3jh.3.
//!
//! Validates that all three contract schema files:
//! 1. Are syntactically valid JSON.
//! 2. Conform to expected JSON Schema structure.
//! 3. Accept well-formed sample documents.
//! 4. Reject structurally invalid documents.
//!
//! Run: `cargo test -p fsci-conformance --test schema_validation`

#![forbid(unsafe_code)]

use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

fn schema_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("docs/schemas")
}

fn load_schema(name: &str) -> Value {
    let path = schema_dir().join(name);
    let raw = fs::read_to_string(&path).unwrap_or_else(|e| {
        assert_failed(&format!("failed to read schema {}: {e}", path.display()))
    });
    serde_json::from_str(&raw).unwrap_or_else(|e| {
        assert_failed(&format!("schema {} is not valid JSON: {e}", path.display()))
    })
}

fn assert_failed(message: &str) -> ! {
    assert!(message.is_empty(), "{message}");
    loop {
        std::hint::spin_loop();
    }
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

fn hardened_case_count(fixture_name: &str) -> Result<usize, String> {
    let path = fixture_path(fixture_name);
    let raw = fs::read_to_string(&path)
        .map_err(|e| format!("failed to read fixture {}: {e}", path.display()))?;
    let fixture: Value = serde_json::from_str(&raw)
        .map_err(|e| format!("fixture {} is not valid JSON: {e}", path.display()))?;
    let cases = fixture
        .get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("fixture {} missing cases array", path.display()))?;
    Ok(cases
        .iter()
        .filter(|case| case.get("mode").and_then(Value::as_str) == Some("Hardened"))
        .count())
}

/// Verify that a JSON Schema document has the required top-level structure.
fn assert_valid_json_schema(schema: &Value, name: &str) {
    let Some(obj) = schema.as_object() else {
        assert_failed(&format!("{name}: schema root must be an object"));
    };

    assert!(obj.contains_key("$schema"), "{name}: missing $schema field");
    assert!(obj.contains_key("title"), "{name}: missing title field");
    assert!(obj.contains_key("type"), "{name}: missing type field");
    assert_eq!(
        obj["type"].as_str().unwrap(),
        "object",
        "{name}: root type must be 'object'"
    );
    assert!(
        obj.contains_key("required"),
        "{name}: missing required field"
    );
    assert!(
        obj.contains_key("properties"),
        "{name}: missing properties field"
    );

    // Verify schema_version is present in properties and constrained to 1
    let props = obj["properties"].as_object().unwrap();
    assert!(
        props.contains_key("schema_version"),
        "{name}: properties must include schema_version"
    );
    let sv = &props["schema_version"];
    assert_eq!(
        sv["const"].as_i64(),
        Some(1),
        "{name}: schema_version must be const: 1"
    );

    // Verify required fields are listed
    let Some(required) = obj["required"].as_array() else {
        assert_failed(&format!("{name}: required must be an array"));
    };
    assert!(
        required
            .iter()
            .any(|v| v.as_str() == Some("schema_version")),
        "{name}: schema_version must be in required"
    );
}

#[test]
fn priority_families_keep_minimum_hardened_fixture_coverage() {
    for fixture_name in [
        "FSCI-P2C-002_linalg_core.json",
        "FSCI-P2C-005_fft_core.json",
        "FSCI-P2C-007_arrayapi_core.json",
        "FSCI-P2C-009_cluster_core.json",
        "FSCI-P2C-011_signal_core.json",
        "FSCI-P2C-012_stats_core.json",
    ] {
        let count = match hardened_case_count(fixture_name) {
            Ok(count) => count,
            Err(message) => {
                assert!(
                    message.is_empty(),
                    "failed to inspect hardened coverage fixture: {message}"
                );
                0
            }
        };
        assert!(
            count >= 5,
            "{fixture_name} must keep at least 5 Hardened-mode cases, got {count}"
        );
    }
}

/// Verify that a schema's $defs entries all have required + properties.
fn assert_defs_well_formed(schema: &Value, name: &str) {
    if let Some(defs) = schema.get("$defs").and_then(|d| d.as_object()) {
        for (def_name, def_value) in defs {
            let Some(def_obj) = def_value.as_object() else {
                assert_failed(&format!(
                    "{name}/$defs/{def_name}: definition must be an object"
                ));
            };
            assert!(
                def_obj.contains_key("type"),
                "{name}/$defs/{def_name}: missing type"
            );
            if def_obj["type"].as_str() == Some("object") {
                assert!(
                    def_obj.contains_key("required"),
                    "{name}/$defs/{def_name}: object definition must have required"
                );
                assert!(
                    def_obj.contains_key("properties"),
                    "{name}/$defs/{def_name}: object definition must have properties"
                );
            }
        }
    }
}

/// Programmatic check: does a sample document match the schema's required fields and types?
fn validate_sample_against_schema(schema: &Value, sample: &Value, name: &str) -> Vec<String> {
    let mut errors = Vec::new();
    let schema_obj = match schema.as_object() {
        Some(o) => o,
        None => {
            errors.push(format!("{name}: schema is not an object"));
            return errors;
        }
    };
    let sample_obj = match sample.as_object() {
        Some(o) => o,
        None => {
            errors.push(format!("{name}: sample is not an object"));
            return errors;
        }
    };

    // Check required fields
    if let Some(required) = schema_obj.get("required").and_then(|r| r.as_array()) {
        for req in required {
            if let Some(field) = req.as_str()
                && !sample_obj.contains_key(field)
            {
                errors.push(format!("{name}: missing required field '{field}'"));
            }
        }
    }

    // Check additionalProperties constraint
    if schema_obj.get("additionalProperties") == Some(&Value::Bool(false))
        && let Some(props) = schema_obj.get("properties").and_then(|p| p.as_object())
    {
        for key in sample_obj.keys() {
            if !props.contains_key(key) {
                errors.push(format!("{name}: unexpected field '{key}'"));
            }
        }
    }

    // Check const values
    if let Some(props) = schema_obj.get("properties").and_then(|p| p.as_object()) {
        for (field, prop_schema) in props {
            if let Some(const_val) = prop_schema.get("const")
                && let Some(actual) = sample_obj.get(field)
                && actual != const_val
            {
                errors.push(format!(
                    "{name}: field '{field}' must be {const_val}, got {actual}"
                ));
            }
        }
    }

    errors
}

// ── Behavior Ledger Schema ──

#[test]
fn behavior_ledger_schema_is_valid_json() {
    let _ = load_schema("behavior_ledger.schema.json");
}

#[test]
fn behavior_ledger_schema_structure() {
    let schema = load_schema("behavior_ledger.schema.json");
    assert_valid_json_schema(&schema, "behavior_ledger");
    assert_defs_well_formed(&schema, "behavior_ledger");

    let props = schema["properties"].as_object().unwrap();
    assert!(props.contains_key("packet_id"), "must have packet_id");
    assert!(props.contains_key("domain"), "must have domain");
    assert!(props.contains_key("entries"), "must have entries");
}

#[test]
fn behavior_ledger_accepts_valid_sample() {
    let schema = load_schema("behavior_ledger.schema.json");
    let sample: Value = serde_json::from_str(
        r#"{
          "schema_version": 1,
          "packet_id": "FSCI-P2C-002",
          "domain": "linalg",
          "entries": [
            {
              "scipy_module": "scipy.linalg._basic",
              "key_functions": ["solve", "inv"],
              "behavior_summary": "Dense linear algebra wrappers over BLAS/LAPACK with error signaling.",
              "rust_strategy": "Reimplement using nalgebra with CASP-driven solver selection.",
              "risk_level": "critical"
            }
          ]
        }"#,
    )
    .unwrap();

    let errors = validate_sample_against_schema(&schema, &sample, "behavior_ledger_valid");
    assert!(errors.is_empty(), "valid sample rejected: {errors:?}");
}

#[test]
fn behavior_ledger_rejects_missing_required() {
    let schema = load_schema("behavior_ledger.schema.json");
    // Missing packet_id and entries
    let sample: Value =
        serde_json::from_str(r#"{"schema_version": 1, "domain": "linalg"}"#).unwrap();

    let errors = validate_sample_against_schema(&schema, &sample, "behavior_ledger_invalid");
    assert!(
        errors.iter().any(|e| e.contains("packet_id")),
        "should reject missing packet_id: {errors:?}"
    );
    assert!(
        errors.iter().any(|e| e.contains("entries")),
        "should reject missing entries: {errors:?}"
    );
}

// ── Contract Table Schema ──

#[test]
fn contract_table_schema_is_valid_json() {
    let _ = load_schema("contract_table.schema.json");
}

#[test]
fn contract_table_schema_structure() {
    let schema = load_schema("contract_table.schema.json");
    assert_valid_json_schema(&schema, "contract_table");
    assert_defs_well_formed(&schema, "contract_table");

    let props = schema["properties"].as_object().unwrap();
    assert!(props.contains_key("packet_id"), "must have packet_id");
    assert!(props.contains_key("domain"), "must have domain");
    assert!(props.contains_key("contracts"), "must have contracts");

    // Verify key defs exist
    let defs = schema["$defs"].as_object().unwrap();
    assert!(
        defs.contains_key("ContractEntry"),
        "must define ContractEntry"
    );
    assert!(defs.contains_key("ParamSpec"), "must define ParamSpec");
    assert!(defs.contains_key("OutputSpec"), "must define OutputSpec");
    assert!(
        defs.contains_key("ErrorCondition"),
        "must define ErrorCondition"
    );
    assert!(
        defs.contains_key("TolerancePolicy"),
        "must define TolerancePolicy"
    );
}

#[test]
fn contract_table_accepts_valid_sample() {
    let schema = load_schema("contract_table.schema.json");
    let sample: Value = serde_json::from_str(
        r#"{
          "schema_version": 1,
          "packet_id": "FSCI-P2C-002",
          "domain": "linalg",
          "contracts": [
            {
              "function_name": "fsci_linalg::solve",
              "inputs": [
                {"name": "a", "type_desc": "2D array f64", "required": true},
                {"name": "b", "type_desc": "1D array f64", "required": true}
              ],
              "outputs": [
                {"name": "x", "type_desc": "Vec<f64>"}
              ],
              "error_conditions": [
                {
                  "condition": "a is singular",
                  "error_type": "SingularMatrix",
                  "message_pattern": "matrix is singular"
                }
              ],
              "tolerance_policy": {
                "comparison_mode": "mixed",
                "default_atol": 1e-12,
                "default_rtol": 1e-12
              },
              "strict_mode_behavior": "Follow SciPy-observable behavior exactly for scoped inputs.",
              "hardened_mode_behavior": "Same contract with bounded condition-number checks and audit logging."
            }
          ]
        }"#,
    )
    .unwrap();

    let errors = validate_sample_against_schema(&schema, &sample, "contract_table_valid");
    assert!(errors.is_empty(), "valid sample rejected: {errors:?}");
}

#[test]
fn contract_table_rejects_missing_required() {
    let schema = load_schema("contract_table.schema.json");
    let sample: Value = serde_json::from_str(r#"{"schema_version": 1}"#).unwrap();

    let errors = validate_sample_against_schema(&schema, &sample, "contract_table_invalid");
    assert!(
        errors.iter().any(|e| e.contains("packet_id")),
        "should reject missing packet_id: {errors:?}"
    );
    assert!(
        errors.iter().any(|e| e.contains("contracts")),
        "should reject missing contracts: {errors:?}"
    );
}

#[test]
fn contract_table_rejects_wrong_schema_version() {
    let schema = load_schema("contract_table.schema.json");
    let sample: Value = serde_json::from_str(
        r#"{
          "schema_version": 99,
          "packet_id": "FSCI-P2C-002",
          "domain": "linalg",
          "contracts": []
        }"#,
    )
    .unwrap();

    let errors = validate_sample_against_schema(&schema, &sample, "contract_table_bad_version");
    assert!(
        errors.iter().any(|e| e.contains("schema_version")),
        "should reject wrong schema_version: {errors:?}"
    );
}

// ── Threat Matrix Schema ──

#[test]
fn threat_matrix_schema_is_valid_json() {
    let _ = load_schema("threat_matrix.schema.json");
}

#[test]
fn threat_matrix_schema_structure() {
    let schema = load_schema("threat_matrix.schema.json");
    assert_valid_json_schema(&schema, "threat_matrix");
    assert_defs_well_formed(&schema, "threat_matrix");

    let props = schema["properties"].as_object().unwrap();
    assert!(props.contains_key("threats"), "must have threats");
    assert!(
        props.contains_key("fail_closed_policies"),
        "must have fail_closed_policies"
    );
    assert!(
        props.contains_key("compatibility_envelope"),
        "must have compatibility_envelope"
    );

    // Verify key defs exist
    let defs = schema["$defs"].as_object().unwrap();
    assert!(defs.contains_key("ThreatEntry"), "must define ThreatEntry");
    assert!(
        defs.contains_key("FailClosedPolicy"),
        "must define FailClosedPolicy"
    );
    assert!(
        defs.contains_key("CompatibilityEnvelope"),
        "must define CompatibilityEnvelope"
    );
}

#[test]
fn threat_matrix_accepts_valid_sample() {
    let schema = load_schema("threat_matrix.schema.json");
    let sample: Value = serde_json::from_str(
        r#"{
          "schema_version": 1,
          "scope": "project",
          "threats": [
            {
              "threat_id": "THREAT-001",
              "subsystem": "fsci-linalg",
              "category": "numerical_instability",
              "severity": "critical",
              "likelihood": "likely",
              "mitigation": "CASP solver portfolio selects SVD fallback for ill-conditioned matrices.",
              "test_reference": "linalg_packet::solve_diagonal_ill_conditioned_warning"
            }
          ],
          "fail_closed_policies": [
            {
              "input_class": "unknown metadata fields",
              "policy": "reject",
              "strict_mode_action": "Return error immediately.",
              "hardened_mode_action": "Return error with diagnostic metadata."
            }
          ]
        }"#,
    )
    .unwrap();

    let errors = validate_sample_against_schema(&schema, &sample, "threat_matrix_valid");
    assert!(errors.is_empty(), "valid sample rejected: {errors:?}");
}

#[test]
fn threat_matrix_rejects_missing_required() {
    let schema = load_schema("threat_matrix.schema.json");
    let sample: Value = serde_json::from_str(r#"{"schema_version": 1}"#).unwrap();

    let errors = validate_sample_against_schema(&schema, &sample, "threat_matrix_invalid");
    assert!(
        errors.iter().any(|e| e.contains("threats")),
        "should reject missing threats: {errors:?}"
    );
    assert!(
        errors.iter().any(|e| e.contains("fail_closed_policies")),
        "should reject missing fail_closed_policies: {errors:?}"
    );
}

#[test]
fn threat_matrix_rejects_additional_properties() {
    let schema = load_schema("threat_matrix.schema.json");
    let sample: Value = serde_json::from_str(
        r#"{
          "schema_version": 1,
          "threats": [],
          "fail_closed_policies": [],
          "unexpected_field": "should be rejected"
        }"#,
    )
    .unwrap();

    let errors = validate_sample_against_schema(&schema, &sample, "threat_matrix_extra_props");
    assert!(
        errors.iter().any(|e| e.contains("unexpected_field")),
        "should reject additional properties: {errors:?}"
    );
}

#[test]
fn p2c001_threat_matrix_artifact_is_schema_aligned_and_complete() {
    let canonical = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/artifacts/FSCI-P2C-001/threats/threat_matrix.json");
    let legacy = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/artifacts/P2C-001/threats/threat_matrix.json");
    let artifact_path = if canonical.exists() {
        canonical
    } else {
        legacy
    };

    let raw = fs::read_to_string(&artifact_path).unwrap_or_else(|error| {
        assert_failed(&format!(
            "failed to read P2C-001 threat artifact {}: {error}",
            artifact_path.display()
        ));
    });
    let artifact: Value = serde_json::from_str(&raw).unwrap_or_else(|error| {
        assert_failed(&format!(
            "failed to parse P2C-001 threat artifact {}: {error}",
            artifact_path.display()
        ));
    });

    let schema = load_schema("threat_matrix.schema.json");
    let errors = validate_sample_against_schema(&schema, &artifact, "p2c001_threat_matrix");
    assert!(
        errors.is_empty(),
        "P2C-001 threat artifact violates schema contract: {errors:?}"
    );

    assert_eq!(artifact["packet_id"], "FSCI-P2C-001");
    assert_eq!(artifact["scope"], "packet");

    let threats = artifact["threats"]
        .as_array()
        .expect("threats must be an array");
    assert!(
        threats.len() >= 8,
        "threat matrix must include >=8 entries, found {}",
        threats.len()
    );

    let mut categories = BTreeSet::new();
    for threat in threats {
        let category = threat["category"]
            .as_str()
            .expect("threat.category must be a string");
        categories.insert(category.to_owned());
        assert!(
            threat["severity"].is_string(),
            "threat.severity must be present"
        );
        assert!(
            threat["likelihood"].is_string(),
            "threat.likelihood must be present"
        );
        assert!(
            threat["mitigation"]
                .as_str()
                .is_some_and(|value| value.len() >= 5),
            "threat.mitigation must be non-trivial"
        );
        assert!(
            threat["test_reference"]
                .as_str()
                .is_some_and(|value| !value.is_empty()),
            "threat.test_reference must be present"
        );
    }

    let expected_categories = BTreeSet::from([
        String::from("compatibility_drift"),
        String::from("malformed_input"),
        String::from("numerical_instability"),
        String::from("resource_exhaustion"),
    ]);
    assert_eq!(
        categories, expected_categories,
        "threat matrix must cover all four required categories"
    );

    assert_eq!(
        artifact["compatibility_envelope"]["scipy_version_min"],
        "1.12.0"
    );
    assert_eq!(
        artifact["compatibility_envelope"]["scipy_version_max"],
        "1.17.0"
    );

    let policies = artifact["fail_closed_policies"]
        .as_array()
        .expect("fail_closed_policies must be an array");
    assert!(
        policies.len() >= 4,
        "fail_closed_policies should include explicit handling for unknown classes"
    );
}

// ── Cross-schema consistency ──

#[test]
fn all_schemas_share_schema_version_contract() {
    let schemas = [
        (
            "behavior_ledger",
            load_schema("behavior_ledger.schema.json"),
        ),
        ("contract_table", load_schema("contract_table.schema.json")),
        ("threat_matrix", load_schema("threat_matrix.schema.json")),
    ];

    for (name, schema) in &schemas {
        let props = schema["properties"].as_object().unwrap();
        let sv = &props["schema_version"];
        assert_eq!(
            sv["type"].as_str(),
            Some("integer"),
            "{name}: schema_version type"
        );
        assert_eq!(
            sv["const"].as_i64(),
            Some(1),
            "{name}: schema_version const"
        );
    }
}

#[test]
fn all_schemas_use_draft_2020_12() {
    let schemas = [
        load_schema("behavior_ledger.schema.json"),
        load_schema("contract_table.schema.json"),
        load_schema("threat_matrix.schema.json"),
    ];

    for schema in &schemas {
        assert_eq!(
            schema["$schema"].as_str(),
            Some("https://json-schema.org/draft/2020-12/schema"),
            "all schemas must use JSON Schema Draft 2020-12"
        );
    }
}

#[test]
fn artifact_topology_document_exists() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("docs/ARTIFACT_TOPOLOGY.md");
    assert!(
        path.exists(),
        "ARTIFACT_TOPOLOGY.md must exist at {}",
        path.display()
    );
    let content = fs::read_to_string(&path).unwrap();
    assert!(
        content.contains("Artifact Topology"),
        "document must have the expected title"
    );
    assert!(
        content.contains("behavior_ledger.schema.json"),
        "document must reference behavior_ledger schema"
    );
    assert!(
        content.contains("contract_table.schema.json"),
        "document must reference contract_table schema"
    );
    assert!(
        content.contains("threat_matrix.schema.json"),
        "document must reference threat_matrix schema"
    );
    assert!(
        content.contains("Governance"),
        "document must include governance section"
    );
}

// ══════════════════════════════════════════════════════════════════════
// Fixture envelope validation (per frankenscipy-vh06)
// ══════════════════════════════════════════════════════════════════════
//
// Loads every FSCI-P2C-*.json fixture under fixtures/ and checks its
// common envelope: packet_id, family, cases array. For each case,
// verifies a case_id is present. Goes beyond the inline-sample tests
// above which only exercise the 3 contract schemas against hand-typed
// JSON — this sweep covers the actual fixtures used by the packet
// runners.

#[test]
fn all_fixtures_have_valid_envelope() {
    let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let mut violations: Vec<String> = Vec::new();
    let mut checked = 0usize;

    let entries = fs::read_dir(&fixtures_dir)
        .unwrap_or_else(|e| assert_failed(&format!("read fixtures dir {fixtures_dir:?}: {e}")));
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !(name.starts_with("FSCI-P2C-") && name.ends_with(".json")) {
            continue;
        }
        // Golden fixtures (FSCI-P2C-*_golden.json) are scipy ppf snapshots
        // with their own envelope (packet_id/family/oracle/points/ppf) and
        // are consumed by golden-runner integration tests, not the packet
        // runners. They don't carry a `cases` array; exempt them from the
        // packet-envelope sweep (frankenscipy-ohv09).
        if name.ends_with("_golden.json") {
            continue;
        }
        let raw = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                violations.push(format!("{name}: read failed: {e}"));
                continue;
            }
        };
        let value: serde_json::Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(e) => {
                violations.push(format!("{name}: JSON parse failed: {e}"));
                continue;
            }
        };
        let obj = match value.as_object() {
            Some(o) => o,
            None => {
                violations.push(format!("{name}: top-level is not an object"));
                continue;
            }
        };
        for req in ["packet_id", "family", "cases"] {
            if !obj.contains_key(req) {
                violations.push(format!("{name}: missing required field `{req}`"));
            }
        }
        // Validate cases array shape if present.
        if let Some(cases) = obj.get("cases").and_then(|v| v.as_array()) {
            for (i, case) in cases.iter().enumerate() {
                let Some(case_obj) = case.as_object() else {
                    violations.push(format!("{name}: cases[{i}] is not an object"));
                    continue;
                };
                if !case_obj.contains_key("case_id") && !case_obj.contains_key("fixture_id") {
                    violations.push(format!(
                        "{name}: cases[{i}] missing `case_id` / `fixture_id`"
                    ));
                }
                // Expected outcome must be present (either `expected` or
                // its legacy names). validate_tol fixtures use `expected`;
                // constants uses `expected_properties`.
                let has_expected = case_obj.contains_key("expected")
                    || case_obj.contains_key("expected_properties");
                if !has_expected {
                    let case_id = case_obj
                        .get("case_id")
                        .or_else(|| case_obj.get("fixture_id"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("<missing>");
                    violations.push(format!(
                        "{name}: cases[{i}] case_id={case_id} missing `expected` or `expected_properties`"
                    ));
                }
            }
        } else if obj.contains_key("cases") {
            violations.push(format!("{name}: `cases` is present but not an array"));
        }
        checked += 1;
    }

    assert!(checked > 0, "no fixtures found under {fixtures_dir:?}");
    assert!(
        violations.is_empty(),
        "fixture envelope violations across {checked} files:\n  - {}",
        violations.join("\n  - ")
    );
    eprintln!("\n── Fixture envelope validation ──\n  {checked} fixtures passed envelope check");
}
