//! Perf topology regression test (frankenscipy-rvox).
//!
//! Verifies that every perf_*.rs file in crates/fsci-conformance/tests/ is
//! included in CI G6's perf gate or explicitly excluded with rationale.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn project_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(root) = manifest_dir.parent().and_then(|p| p.parent()) {
        root.to_path_buf()
    } else {
        manifest_dir
    }
}

fn g6_job_block(workflow: &str) -> Option<&str> {
    let start = workflow.find("  g6-perf:")?;
    let tail = &workflow[start..];
    let end = tail
        .find("\n  # ── G7")
        .or_else(|| tail.find("\n  g7-"))
        .unwrap_or(tail.len());
    Some(&tail[..end])
}

const G6_PERF_TESTS: &[&str] = &[
    "perf_arrayapi",
    "perf_casp",
    "perf_fft",
    "perf_ivp",
    "perf_linalg",
    "perf_optimize",
    "perf_sparse",
    "perf_special",
];

const EXCLUDED_PERF_TESTS: &[&str] = &["perf_topology"];

#[test]
fn every_perf_test_in_g6_or_excluded() -> TestResult {
    let tests_dir = project_root().join("crates/fsci-conformance/tests");

    if !tests_dir.exists() {
        eprintln!("tests/ directory not found, skipping topology check");
        return Ok(());
    }

    let g6_set: HashSet<&str> = G6_PERF_TESTS.iter().copied().collect();
    let excluded_set: HashSet<&str> = EXCLUDED_PERF_TESTS.iter().copied().collect();

    let mut perf_files: Vec<String> = Vec::new();
    for entry in fs::read_dir(&tests_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("perf_") && name.ends_with(".rs") {
            let test_name = name.trim_end_matches(".rs").to_string();
            perf_files.push(test_name);
        }
    }

    let mut missing: Vec<&str> = Vec::new();
    for test_name in &perf_files {
        if !g6_set.contains(test_name.as_str()) && !excluded_set.contains(test_name.as_str()) {
            missing.push(test_name);
        }
    }

    if !missing.is_empty() {
        missing.sort();
        return Err(format!(
            "Perf tests not in CI G6 gate or EXCLUDED_PERF_TESTS:\n  {}\n\n\
             Per bead frankenscipy-rvox, every perf_*.rs file must either be:\n\
             - Listed in G6_PERF_TESTS (runs in CI), or\n\
             - Listed in EXCLUDED_PERF_TESTS with rationale\n\
             Update this file or .github/workflows/ci.yml G6 section.",
            missing.join("\n  ")
        )
        .into());
    }

    Ok(())
}

#[test]
fn g6_tests_all_exist() -> TestResult {
    let tests_dir = project_root().join("crates/fsci-conformance/tests");

    if !tests_dir.exists() {
        eprintln!("tests/ directory not found, skipping existence check");
        return Ok(());
    }

    let mut missing: Vec<&str> = Vec::new();
    for test_name in G6_PERF_TESTS {
        let test_path = tests_dir.join(format!("{}.rs", test_name));
        if !test_path.exists() {
            missing.push(test_name);
        }
    }

    if !missing.is_empty() {
        missing.sort();
        return Err(format!(
            "G6_PERF_TESTS references non-existent test files:\n  {}\n\n\
             Either create the test file or remove it from G6_PERF_TESTS.",
            missing.join("\n  ")
        )
        .into());
    }

    Ok(())
}

#[test]
fn g6_workflow_invokes_all_listed_perf_tests() -> TestResult {
    let workflow_path = project_root().join(".github/workflows/ci.yml");
    let workflow = fs::read_to_string(&workflow_path)?;
    let g6_block = g6_job_block(&workflow).ok_or_else(|| {
        format!(
            "CI workflow {} does not contain a g6-perf job",
            workflow_path.display()
        )
    })?;

    // Since frankenscipy-liel6 the G6 job discovers perf tests with a glob loop
    // (`for f in crates/fsci-conformance/tests/perf_*.rs`) instead of a literal
    // per-test list, so the workflow contract is "the glob loop is present and
    // each discovered file is run via `cargo test --test`". The file-set side of
    // that contract is guarded by `every_perf_test_in_g6_or_excluded` above.
    if !g6_block.contains("crates/fsci-conformance/tests/perf_*.rs") {
        return Err(
            "G6 job no longer runs the perf_*.rs glob loop; every perf test must \
             stay wired into CI G6. Restore the loop or update this topology test \
             to the new invocation shape."
                .into(),
        );
    }
    if !g6_block.contains("cargo test -p fsci-conformance --test") {
        return Err("G6 job no longer runs the discovered perf tests via \
             `cargo test -p fsci-conformance --test`; verify how perf tests are \
             invoked and update this topology test."
            .into());
    }

    Ok(())
}
