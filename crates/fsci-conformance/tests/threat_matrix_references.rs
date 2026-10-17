//! Reference-resolution and schema-conformance gate for per-packet threat matrices
//! (`frankenscipy-3cu8u.6`).
//!
//! Verifies for all 18 packets (`FSCI-P2C-001` .. `FSCI-P2C-018`) and `docs/threat_matrix.json`:
//! 1. Every packet has `fixtures/artifacts/FSCI-P2C-XXX/threats/threat_matrix.json`.
//! 2. Every matrix strictly conforms to `docs/schemas/threat_matrix.schema.json`
//!    (`additionalProperties: false`, required fields, enums, `THREAT-NNN` pattern).
//! 3. No two distinct `FSCI-P2C-XXX` packets share identical threat matrices, and each packet's
//!    `subsystem` entries match that packet's crate domain (catching the P2C-007 copy of P2C-001).
//! 4. Every defended threat entry names a real defending function `crates/.../file.rs:line (fn_name)`
//!    in `mitigation` whose file exists, line is in range, and function definition exists in that
//!    file (unless explicitly marked `status: undefended`).
//! 5. Every threat entry's `test_reference` resolves to an existing `#[test]` function in the
//!    workspace or an existing fixture `case_id` in `crates/fsci-conformance/fixtures/*.json`.

#![forbid(unsafe_code)]

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

fn conformance_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn expected_packet_subsystem(packet_id: &str) -> &'static str {
    match packet_id {
        "FSCI-P2C-001" => "fsci-integrate",
        "FSCI-P2C-002" => "fsci-linalg",
        "FSCI-P2C-003" => "fsci-opt",
        "FSCI-P2C-004" => "fsci-sparse",
        "FSCI-P2C-005" => "fsci-fft",
        "FSCI-P2C-006" => "fsci-special",
        "FSCI-P2C-007" => "fsci-arrayapi",
        "FSCI-P2C-008" => "fsci-runtime",
        "FSCI-P2C-009" => "fsci-cluster",
        "FSCI-P2C-010" => "fsci-spatial",
        "FSCI-P2C-011" => "fsci-signal",
        "FSCI-P2C-012" => "fsci-stats",
        "FSCI-P2C-013" => "fsci-integrate",
        "FSCI-P2C-014" => "fsci-interpolate",
        "FSCI-P2C-015" => "fsci-ndimage",
        "FSCI-P2C-016" => "fsci-constants",
        "FSCI-P2C-017" => "fsci-io",
        "FSCI-P2C-018" => "fsci-odr",
        _ => panic!("unknown packet_id: {packet_id}"),
    }
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn collect_workspace_test_names(root: &Path) -> BTreeSet<String> {
    let mut rs_files = Vec::new();
    collect_rs_files(&root.join("crates"), &mut rs_files);
    let mut tests = BTreeSet::new();
    for file in rs_files {
        let Ok(content) = fs::read_to_string(&file) else {
            continue;
        };
        let lines: Vec<&str> = content.lines().collect();
        for (idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with("#[test]") || trimmed.ends_with("::test]") {
                for follow in lines.iter().skip(idx + 1).take(8) {
                    let ftrim = follow.trim();
                    if let Some(rest) = ftrim
                        .strip_prefix("fn ")
                        .or_else(|| ftrim.strip_prefix("pub fn "))
                        .or_else(|| ftrim.strip_prefix("async fn "))
                    {
                        let name: String = rest
                            .chars()
                            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                            .collect();
                        if !name.is_empty() {
                            tests.insert(name);
                        }
                        break;
                    }
                }
            }
        }
    }
    tests
}

fn collect_fixture_cases(conf_dir: &Path) -> BTreeMap<String, BTreeSet<String>> {
    let mut map = BTreeMap::new();
    let fixtures_dir = conf_dir.join("fixtures");
    let Ok(entries) = fs::read_dir(&fixtures_dir) else {
        return map;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(fname) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !fname.ends_with(".json") {
            continue;
        }
        let Ok(raw) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(val) = serde_json::from_str::<Value>(&raw) else {
            continue;
        };
        let mut cases = BTreeSet::new();
        if let Some(arr) = val.get("cases").and_then(Value::as_array) {
            for c in arr {
                if let Some(cid) = c
                    .get("case_id")
                    .or_else(|| c.get("fixture_id"))
                    .and_then(Value::as_str)
                {
                    cases.insert(cid.to_string());
                }
            }
        }
        map.insert(fname.to_string(), cases);
    }
    map
}

fn is_threat_id_valid(id: &str) -> bool {
    let Some(suffix) = id.strip_prefix("THREAT-") else {
        return false;
    };
    suffix.len() == 3 && suffix.chars().all(|c| c.is_ascii_digit())
}

fn parse_defending_fn_ref(mitigation: &str) -> Option<(String, usize, String)> {
    // Expect pattern `crates/<crate>/src/<path>.rs:<line> (<fn_name>)` inside mitigation.
    let start = mitigation.find("crates/")?;
    let sub = &mitigation[start..];
    let colon_pos = sub.find(".rs:")?;
    let file_rel = &sub[..colon_pos + 3];
    let after_colon = &sub[colon_pos + 4..];
    let digits: String = after_colon
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    let line_no: usize = digits.parse().ok()?;
    let after_digits = &after_colon[digits.len()..];
    let open_paren = after_digits.find('(')?;
    let close_paren = after_digits[open_paren + 1..].find(')')?;
    let fn_name = after_digits[open_paren + 1..open_paren + 1 + close_paren].trim();
    if fn_name.is_empty() {
        return None;
    }
    Some((file_rel.to_string(), line_no, fn_name.to_string()))
}

fn validate_matrix_document(
    label: &str,
    expected_packet_id: Option<&str>,
    doc: &Value,
    root: &Path,
    all_tests: &BTreeSet<String>,
    fixture_cases: &BTreeMap<String, BTreeSet<String>>,
) -> Vec<String> {
    let mut errors = Vec::new();
    let Some(obj) = doc.as_object() else {
        errors.push(format!("{label}: root is not a JSON object"));
        return errors;
    };

    let allowed_top = [
        "schema_version",
        "packet_id",
        "scope",
        "compatibility_envelope",
        "generated_at",
        "threats",
        "fail_closed_policies",
    ];
    for k in obj.keys() {
        if !allowed_top.contains(&k.as_str()) {
            errors.push(format!("{label}: unexpected top-level property '{k}'"));
        }
    }
    for req in ["schema_version", "threats", "fail_closed_policies"] {
        if !obj.contains_key(req) {
            errors.push(format!("{label}: missing required top-level field '{req}'"));
        }
    }
    if obj.get("schema_version").and_then(Value::as_i64) != Some(1) {
        errors.push(format!("{label}: schema_version must be 1"));
    }

    if let Some(exp_pid) = expected_packet_id {
        if obj.get("packet_id").and_then(Value::as_str) != Some(exp_pid) {
            errors.push(format!(
                "{label}: packet_id {:?} != expected {exp_pid}",
                obj.get("packet_id")
            ));
        }
        if obj.get("scope").and_then(Value::as_str) != Some("packet") {
            errors.push(format!("{label}: scope must be 'packet'"));
        }
    }

    let raw_str = serde_json::to_string(doc).unwrap_or_default();
    if raw_str.contains("Naive O(n^2) DFT") || raw_str.contains("implement Cooley-Tukey") {
        errors.push(format!(
            "{label}: contains stale unimplemented claim ('Naive O(n^2) DFT' / 'implement Cooley-Tukey')"
        ));
    }

    let Some(threats) = obj.get("threats").and_then(Value::as_array) else {
        errors.push(format!("{label}: 'threats' must be an array"));
        return errors;
    };
    if threats.is_empty() {
        errors.push(format!("{label}: 'threats' must not be empty"));
    }

    let allowed_threat_keys = [
        "threat_id",
        "subsystem",
        "category",
        "severity",
        "likelihood",
        "description",
        "mitigation",
        "test_reference",
        "strict_mode_response",
        "hardened_mode_response",
        "required_artifact",
    ];
    let allowed_categories = [
        "numerical_instability",
        "malformed_input",
        "resource_exhaustion",
        "compatibility_drift",
    ];
    let allowed_severities = ["critical", "high", "medium", "low"];
    let allowed_likelihoods = ["certain", "likely", "possible", "unlikely", "rare"];

    for (idx, t) in threats.iter().enumerate() {
        let Some(tobj) = t.as_object() else {
            errors.push(format!("{label}: threats[{idx}] is not an object"));
            continue;
        };
        let tid = tobj
            .get("threat_id")
            .and_then(Value::as_str)
            .unwrap_or("<missing-threat_id>");
        if !is_threat_id_valid(tid) {
            errors.push(format!(
                "{label}:{tid}: threat_id must match ^THREAT-\\d{{3}}$"
            ));
        }
        for k in tobj.keys() {
            if !allowed_threat_keys.contains(&k.as_str()) {
                errors.push(format!("{label}:{tid}: unexpected property '{k}'"));
            }
        }
        for req in [
            "threat_id",
            "subsystem",
            "category",
            "severity",
            "likelihood",
            "mitigation",
            "test_reference",
        ] {
            if !tobj.contains_key(req) {
                errors.push(format!("{label}:{tid}: missing required field '{req}'"));
            }
        }

        if let Some(sub) = tobj.get("subsystem").and_then(Value::as_str)
            && let Some(exp_pid) = expected_packet_id
        {
            let exp_sub = expected_packet_subsystem(exp_pid);
            if !sub.starts_with(exp_sub) {
                errors.push(format!(
                    "{label}:{tid}: subsystem '{sub}' does not match packet crate '{exp_sub}'"
                ));
            }
        }
        if let Some(cat) = tobj.get("category").and_then(Value::as_str)
            && !allowed_categories.contains(&cat)
        {
            errors.push(format!("{label}:{tid}: invalid category '{cat}'"));
        }
        if let Some(sev) = tobj.get("severity").and_then(Value::as_str)
            && !allowed_severities.contains(&sev)
        {
            errors.push(format!("{label}:{tid}: invalid severity '{sev}'"));
        }
        if let Some(lik) = tobj.get("likelihood").and_then(Value::as_str)
            && !allowed_likelihoods.contains(&lik)
        {
            errors.push(format!("{label}:{tid}: invalid likelihood '{lik}'"));
        }

        if let Some(mit) = tobj.get("mitigation").and_then(Value::as_str) {
            if !mit.contains("status: undefended") {
                match parse_defending_fn_ref(mit) {
                    None => errors.push(format!(
                        "{label}:{tid}: defended row mitigation must cite 'crates/<crate>/src/<file>.rs:<line> (<fn_name>)' or 'status: undefended'"
                    )),
                    Some((file_rel, line_no, fn_name)) => {
                        let fpath = root.join(&file_rel);
                        match fs::read_to_string(&fpath) {
                            Err(_) => errors.push(format!(
                                "{label}:{tid}: defending file '{file_rel}' does not exist"
                            )),
                            Ok(content) => {
                                let line_count = content.lines().count();
                                if line_no == 0 || line_no > line_count {
                                    errors.push(format!(
                                        "{label}:{tid}: line {line_no} out of range (1..={line_count}) in '{file_rel}'"
                                    ));
                                }
                                let fn_pat1 = format!("fn {fn_name}(");
                                let fn_pat2 = format!("fn {fn_name}<");
                                if !content.contains(&fn_pat1) && !content.contains(&fn_pat2) {
                                    errors.push(format!(
                                        "{label}:{tid}: defending function '{fn_name}' not found in '{file_rel}'"
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some(tref) = tobj.get("test_reference").and_then(Value::as_str) {
            let resolved = if let Some((fixture_file, case_id)) = tref.split_once("::") {
                if fixture_file.ends_with(".json") {
                    fixture_cases
                        .get(fixture_file)
                        .is_some_and(|cases| cases.contains(case_id))
                } else {
                    all_tests.contains(case_id)
                }
            } else {
                all_tests.contains(tref)
            };
            if !resolved {
                errors.push(format!("{label}:{tid}: unresolved test_reference '{tref}'"));
            }
        }
    }

    if let Some(policies) = obj.get("fail_closed_policies").and_then(Value::as_array) {
        if policies.is_empty() {
            errors.push(format!("{label}: 'fail_closed_policies' must not be empty"));
        }
        let allowed_policy_keys = [
            "input_class",
            "policy",
            "strict_mode_action",
            "hardened_mode_action",
            "override_mechanism",
        ];
        let allowed_policy_vals = ["reject", "reject_with_diagnostics", "allowlist_only"];
        for (idx, p) in policies.iter().enumerate() {
            let Some(pobj) = p.as_object() else {
                errors.push(format!(
                    "{label}: fail_closed_policies[{idx}] not an object"
                ));
                continue;
            };
            for k in pobj.keys() {
                if !allowed_policy_keys.contains(&k.as_str()) {
                    errors.push(format!(
                        "{label}: fail_closed_policies[{idx}] unexpected key '{k}'"
                    ));
                }
            }
            for req in [
                "input_class",
                "policy",
                "strict_mode_action",
                "hardened_mode_action",
            ] {
                if !pobj.contains_key(req) {
                    errors.push(format!(
                        "{label}: fail_closed_policies[{idx}] missing '{req}'"
                    ));
                }
            }
            if let Some(pol) = pobj.get("policy").and_then(Value::as_str)
                && !allowed_policy_vals.contains(&pol)
            {
                errors.push(format!(
                    "{label}: fail_closed_policies[{idx}] invalid policy '{pol}'"
                ));
            }
        }
    }

    errors
}

#[test]
fn all_18_packet_threat_matrices_and_project_matrix_resolve_to_code_and_tests() {
    let root = workspace_root();
    let conf_dir = conformance_dir();
    let all_tests = collect_workspace_test_names(&root);
    let fixture_cases = collect_fixture_cases(&conf_dir);

    let mut all_errors = Vec::new();
    let mut seen_threats_json: BTreeMap<String, String> = BTreeMap::new();

    for idx in 1..=18 {
        let packet_id = format!("FSCI-P2C-{idx:03}");
        let matrix_path = conf_dir
            .join("fixtures/artifacts")
            .join(&packet_id)
            .join("threats/threat_matrix.json");
        if !matrix_path.exists() {
            all_errors.push(format!(
                "{packet_id}: missing threat matrix at {}",
                matrix_path.display()
            ));
            continue;
        }
        let raw = match fs::read_to_string(&matrix_path) {
            Ok(s) => s,
            Err(e) => {
                all_errors.push(format!("{packet_id}: read failed: {e}"));
                continue;
            }
        };
        let doc: Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(e) => {
                all_errors.push(format!("{packet_id}: invalid JSON: {e}"));
                continue;
            }
        };

        let threats_canonical = serde_json::to_string(&doc["threats"]).unwrap_or_default();
        if let Some(prev_pid) = seen_threats_json.insert(threats_canonical, packet_id.clone()) {
            all_errors.push(format!(
                "{packet_id}: shares identical threats array with {prev_pid}"
            ));
        }

        let errs = validate_matrix_document(
            &packet_id,
            Some(&packet_id),
            &doc,
            &root,
            &all_tests,
            &fixture_cases,
        );
        all_errors.extend(errs);

        // If legacy P2C-XXX twin directory exists, keep its threat_matrix.json synchronized.
        let legacy_id = format!("P2C-{idx:03}");
        let legacy_dir = conf_dir.join("fixtures/artifacts").join(&legacy_id);
        let legacy_matrix = legacy_dir.join("threats/threat_matrix.json");
        if legacy_matrix.exists() {
            let legacy_raw = fs::read_to_string(&legacy_matrix).unwrap_or_default();
            if legacy_raw != raw {
                all_errors.push(format!(
                    "{legacy_id}: legacy threat_matrix.json drifted from {packet_id}"
                ));
            }
        }
    }

    let project_matrix_path = root.join("docs/threat_matrix.json");
    let project_raw =
        fs::read_to_string(&project_matrix_path).expect("read docs/threat_matrix.json");
    let project_doc: Value =
        serde_json::from_str(&project_raw).expect("parse docs/threat_matrix.json");
    all_errors.extend(validate_matrix_document(
        "docs/threat_matrix.json",
        None,
        &project_doc,
        &root,
        &all_tests,
        &fixture_cases,
    ));

    if !all_errors.is_empty() {
        for err in &all_errors {
            eprintln!("UNRESOLVED THREAT MATRIX REFERENCE: {err}");
        }
        panic!(
            "threat_matrix_references failed with {} error(s):\n  - {}",
            all_errors.len(),
            all_errors.join("\n  - ")
        );
    }
}

#[test]
fn negative_controls_reject_p2c007_integrate_copy_and_p2c005_stale_dft() {
    let root = workspace_root();
    let conf_dir = conformance_dir();
    let all_tests = collect_workspace_test_names(&root);
    let fixture_cases = collect_fixture_cases(&conf_dir);

    let p2c007_copy: Value = serde_json::json!({
        "schema_version": 1,
        "packet_id": "FSCI-P2C-007",
        "scope": "packet",
        "threats": [{
            "threat_id": "THREAT-001",
            "subsystem": "fsci-integrate",
            "category": "numerical_instability",
            "severity": "critical",
            "likelihood": "likely",
            "mitigation": "Detect repeated rejections in solve_ivp.",
            "test_reference": "P2C-001/threats/test_stiff_step_collapse.json"
        }],
        "fail_closed_policies": [{
            "input_class": "non-finite",
            "policy": "reject",
            "strict_mode_action": "reject",
            "hardened_mode_action": "reject"
        }]
    });
    let errs_007 = validate_matrix_document(
        "FSCI-P2C-007",
        Some("FSCI-P2C-007"),
        &p2c007_copy,
        &root,
        &all_tests,
        &fixture_cases,
    );
    assert!(
        errs_007.iter().any(|e| e
            .contains("subsystem 'fsci-integrate' does not match packet crate 'fsci-arrayapi'")),
        "expected P2C-007 wrong-subsystem error, got: {errs_007:?}"
    );
    assert!(
        errs_007
            .iter()
            .any(|e| e.contains("unresolved test_reference")),
        "expected P2C-007 unresolved test_reference error, got: {errs_007:?}"
    );

    let p2c005_stale: Value = serde_json::json!({
        "schema_version": 1,
        "packet_id": "FSCI-P2C-005",
        "scope": "packet",
        "threats": [{
            "threat_id": "THREAT-001",
            "subsystem": "fsci-fft",
            "category": "resource_exhaustion",
            "severity": "medium",
            "likelihood": "likely",
            "description": "Naive O(n^2) DFT implementation becomes extremely slow.",
            "mitigation": "Future: implement Cooley-Tukey FFT for O(n log n) performance.",
            "test_reference": "N/A (performance characteristic, not correctness bug)"
        }],
        "fail_closed_policies": [{
            "input_class": "empty",
            "policy": "reject",
            "strict_mode_action": "reject",
            "hardened_mode_action": "reject"
        }]
    });
    let errs_005 = validate_matrix_document(
        "FSCI-P2C-005",
        Some("FSCI-P2C-005"),
        &p2c005_stale,
        &root,
        &all_tests,
        &fixture_cases,
    );
    assert!(
        errs_005
            .iter()
            .any(|e| e.contains("stale unimplemented claim")),
        "expected P2C-005 stale DFT claim error, got: {errs_005:?}"
    );
}
