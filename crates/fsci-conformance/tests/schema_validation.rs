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
use std::fs;
use std::path::PathBuf;

fn schema_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("docs/schemas")
}

fn load_schema(name: &str) -> Value {
    let path = schema_dir().join(name);
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read schema {}: {e}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("schema {} is not valid JSON: {e}", path.display()))
}

/// Verify that a JSON Schema document has the required top-level structure.
fn assert_valid_json_schema(schema: &Value, name: &str) {
    let obj = schema.as_object().unwrap_or_else(|| {
        panic!("{name}: schema root must be an object");
    });

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
    let required = obj["required"]
        .as_array()
        .unwrap_or_else(|| panic!("{name}: required must be an array"));
    assert!(
        required
            .iter()
            .any(|v| v.as_str() == Some("schema_version")),
        "{name}: schema_version must be in required"
    );
}

/// Verify that a schema's $defs entries all have required + properties.
fn assert_defs_well_formed(schema: &Value, name: &str) {
    if let Some(defs) = schema.get("$defs").and_then(|d| d.as_object()) {
        for (def_name, def_value) in defs {
            let def_obj = def_value.as_object().unwrap_or_else(|| {
                panic!("{name}/$defs/{def_name}: definition must be an object");
            });
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
